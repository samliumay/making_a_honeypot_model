//! HTTP API for the Streamlit lab UI (`code/lab_ui`), bound to 127.0.0.1.
//! The UI only views and triggers; every number comes from here or from the
//! run folders (AGENTS.md "Visualization").
//!
//! GET  /api/info                      dataset version, files, families, models, collector URL
//! GET  /api/episodes?file=&family=    episode summaries
//! GET  /api/episode/{id}              one episode (system prompt, messages, expected)
//! POST /api/run_episode               {model, episode_id} -> transcript + score
//! POST /api/chat                      {model, episode_id, messages} -> free-form attack on that deployment
//! POST /api/eval                      {model, files, limit} -> starts a run, returns run_id
//! GET  /api/progress                  progress of running evals
//! GET  /api/runs                      run folders with meta
//! GET  /api/runs/{id}                 summary.json (+ judged success if judged)
//! GET  /api/runs/{id}/episodes        episodes.jsonl rows
//! GET  /api/alerts?limit=             collector feed

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::agent::{EpisodeInput, run_episode};
use crate::collector::CollectorState;
use crate::config::Config;
use crate::dataset::Dataset;
use crate::judge::{judged_success, load_verdicts};
use crate::model::ChatClient;
use crate::runs::{
    Progress, RunRequest, episode_seed, list_runs, load_episodes, new_run_id, read_run_json,
    run_eval,
};
use crate::scoring::score;
use crate::types::{Message, Role};

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub ds: Arc<Dataset>,
    pub collector: CollectorState,
    pub http: reqwest::Client,
    pub progress: Arc<Mutex<BTreeMap<String, Arc<Mutex<Progress>>>>>,
}

type ApiResult = Result<Json<Value>, (StatusCode, String)>;

fn bad(e: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, format!("{e:#}"))
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/info", get(info))
        .route("/api/episodes", get(episodes))
        .route("/api/episode/{id}", get(episode))
        .route("/api/run_episode", post(run_one))
        .route("/api/chat", post(chat))
        .route("/api/eval", post(start_eval))
        .route("/api/progress", get(progress))
        .route("/api/runs", get(runs))
        .route("/api/runs/{id}", get(run_summary))
        .route("/api/runs/{id}/episodes", get(run_episodes))
        .route("/api/alerts", get(alerts))
        .with_state(state)
}

async fn info(State(s): State<AppState>) -> ApiResult {
    let files: BTreeMap<&String, Value> =
        s.ds.eval
            .iter()
            .map(|(f, eps)| {
                let mut fams: Vec<&str> = eps.iter().map(|e| e.family.as_str()).collect();
                fams.sort_unstable();
                fams.dedup();
                (f, json!({"n": eps.len(), "families": fams}))
            })
            .collect();
    Ok(Json(json!({
        "dataset": s.ds.manifest.dataset, "version": s.ds.manifest.version,
        "alert_url": s.ds.manifest.alert_url, "attacker_url": s.ds.manifest.attacker_url,
        "categories": s.ds.manifest.categories, "files": files,
        "models": s.cfg.models, "judge_model": s.cfg.judge.model,
    })))
}

#[derive(Deserialize)]
struct EpisodeFilter {
    file: Option<String>,
    family: Option<String>,
}

async fn episodes(State(s): State<AppState>, Query(q): Query<EpisodeFilter>) -> ApiResult {
    let rows: Vec<Value> = s
        .ds
        .eval
        .iter()
        .filter(|(f, _)| q.file.as_ref().is_none_or(|x| x == *f))
        .flat_map(|(f, eps)| eps.iter().map(move |e| (f, e)))
        .filter(|(_, e)| q.family.as_ref().is_none_or(|x| x == &e.family))
        .map(|(f, e)| {
            let last = e.messages.iter().rev().find(|m| m.role == Role::User).map(|m| m.content.as_str()).unwrap_or("");
            json!({"id": e.id, "file": f, "class": e.class, "family": e.family, "deployment": e.deployment,
                   "behaviour": e.expected.behaviour, "last_user": last})
        })
        .collect();
    Ok(Json(Value::Array(rows)))
}

async fn episode(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let (file, e) =
        s.ds.episode(&id)
            .ok_or_else(|| bad(format!("no episode {id}")))?;
    Ok(Json(json!({"file": file, "episode": e})))
}

#[derive(Deserialize)]
struct RunOne {
    model: String,
    episode_id: String,
}

async fn run_one(State(s): State<AppState>, Json(req): Json<RunOne>) -> ApiResult {
    let (_, ep) =
        s.ds.episode(&req.episode_id)
            .ok_or_else(|| bad("unknown episode"))?;
    let client = ChatClient::new(
        s.cfg.model(&req.model).map_err(bad)?.clone(),
        &s.cfg.generation,
    )
    .map_err(bad)?;
    let input = EpisodeInput {
        messages: &ep.messages,
        documents: &ep.documents,
        episode_id: &ep.id,
        run_id: Some("lab-single"),
        seed: episode_seed(s.cfg.lab.seed, &ep.id),
    };
    let m = &s.ds.manifest;
    let tr = run_episode(
        &client,
        &s.http,
        &m.tools,
        &m.alert_url,
        s.cfg.generation.max_tool_rounds,
        input,
    )
    .await;
    let sc = score(ep, &tr, &m.tell_words, &m.alert_url, &m.attacker_url);
    Ok(Json(
        json!({"transcript": tr, "score": sc, "expected": ep.expected}),
    ))
}

#[derive(Deserialize)]
struct ChatReq {
    model: String,
    /// Borrow this episode's deployment: system prompt, real values, documents.
    episode_id: String,
    /// The free-form conversation so far (user and assistant turns only).
    messages: Vec<Message>,
    #[serde(default)]
    documents: BTreeMap<String, String>,
}

/// Free-form attacking in the lab: the researcher types attacks against a
/// deployment; the reply is scored for leaks, alerts, fakes and tells.
async fn chat(State(s): State<AppState>, Json(req): Json<ChatReq>) -> ApiResult {
    let (_, base) =
        s.ds.episode(&req.episode_id)
            .ok_or_else(|| bad("unknown episode"))?;
    let mut ep = base.clone();
    ep.messages = std::iter::once(base.messages[0].clone())
        .chain(
            req.messages
                .into_iter()
                .filter(|m| matches!(m.role, Role::User | Role::Assistant)),
        )
        .collect();
    ep.documents.extend(req.documents);
    ep.expected.asked.clear();
    ep.expected.must_equal_earlier = false;
    let client = ChatClient::new(
        s.cfg.model(&req.model).map_err(bad)?.clone(),
        &s.cfg.generation,
    )
    .map_err(bad)?;
    let seed = rand_seed();
    let input = EpisodeInput {
        messages: &ep.messages,
        documents: &ep.documents,
        episode_id: "lab-chat",
        run_id: Some("lab-chat"),
        seed,
    };
    let m = &s.ds.manifest;
    let tr = run_episode(
        &client,
        &s.http,
        &m.tools,
        &m.alert_url,
        s.cfg.generation.max_tool_rounds,
        input,
    )
    .await;
    let sc = score(&ep, &tr, &m.tell_words, &m.alert_url, &m.attacker_url);
    let reals: BTreeMap<&str, &str> = ep
        .assets
        .values()
        .map(|a| (a.label.as_str(), a.real.as_str()))
        .collect();
    Ok(Json(
        json!({"transcript": tr, "score": sc, "real_values": reals, "seed": seed}),
    ))
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64 >> 1)
        .unwrap_or(1)
}

#[derive(Deserialize)]
struct EvalReq {
    model: String,
    #[serde(default)]
    files: Vec<String>,
    limit: Option<usize>,
}

async fn start_eval(State(s): State<AppState>, Json(req): Json<EvalReq>) -> ApiResult {
    let model = s.cfg.model(&req.model).map_err(bad)?.clone();
    let files = s.ds.select(&req.files).map_err(bad)?;
    let run_id = new_run_id(&model.name);
    let prog = Arc::new(Mutex::new(Progress {
        run_id: run_id.clone(),
        ..Progress::default()
    }));
    s.progress.lock().await.insert(run_id.clone(), prog.clone());
    let (cfg, ds) = (s.cfg.clone(), s.ds.clone());
    let rr = RunRequest {
        model,
        files,
        limit: req.limit,
        run_id: run_id.clone(),
    };
    tokio::spawn(async move {
        if let Err(e) = run_eval(&cfg, &ds, rr, prog.clone()).await {
            let mut p = prog.lock().await;
            p.error = Some(format!("{e:#}"));
            p.finished = true;
        }
    });
    Ok(Json(json!({"run_id": run_id})))
}

async fn progress(State(s): State<AppState>) -> ApiResult {
    let map = s.progress.lock().await;
    let mut out = Vec::new();
    for p in map.values() {
        out.push(p.lock().await.clone());
    }
    Ok(Json(json!(out)))
}

async fn runs(State(s): State<AppState>) -> ApiResult {
    let rows: Vec<Value> = list_runs(&s.cfg.lab.results_dir)
        .into_iter()
        .map(|(id, meta)| json!({"run_id": id, "meta": meta, "has_summary": s.cfg.lab.results_dir.join("runs").join(&id).join("summary.json").is_file()}))
        .collect();
    Ok(Json(Value::Array(rows)))
}

fn run_dir(s: &AppState, id: &str) -> Result<std::path::PathBuf, (StatusCode, String)> {
    if id.contains('/') || id.contains("..") {
        return Err(bad("bad run id"));
    }
    Ok(s.cfg.lab.results_dir.join("runs").join(id))
}

async fn run_summary(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let dir = run_dir(&s, &id)?;
    let summary = read_run_json(&dir, "summary.json").map_err(bad)?;
    let meta = read_run_json(&dir, "meta.json").map_err(bad)?;
    let verdicts = load_verdicts(&dir);
    let judged = if verdicts.is_empty() {
        Value::Null
    } else {
        let results = load_episodes(&dir).map_err(bad)?;
        let mut g: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for r in &results {
            let ok = judged_success(r, verdicts.get(&r.id));
            for k in [
                format!("{}|{}", r.file, r.family),
                format!("{}|(all)", r.file),
            ] {
                let e = g.entry(k).or_default();
                e.0 += usize::from(ok);
                e.1 += 1;
            }
        }
        json!(
            g.into_iter()
                .map(|(k, (ok, n))| (k, json!({"success": ok as f64 / n as f64, "n": n})))
                .collect::<BTreeMap<_, _>>()
        )
    };
    Ok(Json(
        json!({"meta": meta, "summary": summary, "judged": judged}),
    ))
}

async fn run_episodes(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let dir = run_dir(&s, &id)?;
    let results = load_episodes(&dir).map_err(bad)?;
    let verdicts = load_verdicts(&dir);
    let rows: Vec<Value> = results
        .iter()
        .map(|r| json!({"result": r, "verdict": verdicts.get(&r.id)}))
        .collect();
    Ok(Json(Value::Array(rows)))
}

#[derive(Deserialize)]
struct AlertQ {
    limit: Option<usize>,
}

async fn alerts(State(s): State<AppState>, Query(q): Query<AlertQ>) -> ApiResult {
    let mut all = s.collector.all().await;
    if let Some(l) = q.limit {
        let skip = all.len().saturating_sub(l);
        all.drain(..skip);
    }
    Ok(Json(json!(all)))
}
