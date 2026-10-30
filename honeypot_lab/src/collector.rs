//! The defender's collector: the endpoint the honeypot model learned to alert.
//!
//! It is a real HTTP listener on `collector_addr` (127.0.0.1:8090), so the
//! alert path is the same as it would be outside the lab: the sandbox POSTs the
//! model's request to it. Every alert is kept in memory for the UI and appended
//! to `results/collector/alerts.jsonl`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

/// Header the sandbox adds so an alert can be tied to its episode.
pub const EPISODE_HEADER: &str = "x-p9-episode";
pub const RUN_HEADER: &str = "x-p9-run";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub seq: u64,
    pub received_at: String,
    pub run_id: Option<String>,
    pub episode_id: Option<String>,
    pub body: Value,
}

#[derive(Clone)]
pub struct CollectorState {
    alerts: Arc<Mutex<Vec<Alert>>>,
    log_path: PathBuf,
}

impl CollectorState {
    pub fn new(results_dir: &std::path::Path) -> CollectorState {
        CollectorState {
            alerts: Arc::new(Mutex::new(Vec::new())),
            log_path: results_dir.join("collector").join("alerts.jsonl"),
        }
    }

    pub async fn all(&self) -> Vec<Alert> {
        self.alerts.lock().await.clone()
    }
}

#[derive(Deserialize)]
struct AlertQuery {
    run_id: Option<String>,
    episode_id: Option<String>,
    limit: Option<usize>,
}

pub fn router(state: CollectorState) -> Router {
    Router::new()
        .route("/alert", post(receive))
        .route("/alerts", get(list))
        .with_state(state)
}

async fn receive(
    State(state): State<CollectorState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let mut alerts = state.alerts.lock().await;
    let alert = Alert {
        seq: alerts.len() as u64 + 1,
        received_at: chrono::Utc::now().to_rfc3339(),
        run_id: header(RUN_HEADER),
        episode_id: header(EPISODE_HEADER),
        body,
    };
    if let Err(e) = append_line(&state.log_path, &alert).await {
        eprintln!(
            "collector: could not write {}: {e}",
            state.log_path.display()
        );
    }
    alerts.push(alert);
    Json(serde_json::json!({"status": "ok"}))
}

async fn list(
    State(state): State<CollectorState>,
    Query(q): Query<AlertQuery>,
) -> Json<Vec<Alert>> {
    let alerts = state.alerts.lock().await;
    let mut out: Vec<Alert> = alerts
        .iter()
        .filter(|a| {
            q.run_id
                .as_ref()
                .is_none_or(|r| a.run_id.as_ref() == Some(r))
        })
        .filter(|a| {
            q.episode_id
                .as_ref()
                .is_none_or(|e| a.episode_id.as_ref() == Some(e))
        })
        .cloned()
        .collect();
    if let Some(limit) = q.limit {
        let skip = out.len().saturating_sub(limit);
        out.drain(..skip);
    }
    Json(out)
}

async fn append_line(path: &std::path::Path, alert: &Alert) -> Result<()> {
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    f.write_all(format!("{}\n", serde_json::to_string(alert)?).as_bytes())
        .await?;
    Ok(())
}

/// Start the collector on `addr`. If the port is already taken (the lab server
/// is running and an eval was started separately), the existing collector is
/// used, which is fine because the alert endpoint is the same.
pub async fn spawn(addr: &str, state: CollectorState) -> Result<bool> {
    match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => {
            let app = router(state);
            tokio::spawn(async move {
                if let Err(e) = axum::serve(listener, app).await {
                    eprintln!("collector stopped: {e}");
                }
            });
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Ok(false),
        Err(e) => Err(e.into()),
    }
}
