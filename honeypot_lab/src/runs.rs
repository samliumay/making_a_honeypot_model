//! Eval runs: run every selected episode against one model, score it, and
//! write the run folder `results/runs/<run_id>/`:
//!   meta.json         model, endpoint, dataset version + sha256, seed, commit, timing
//!   config.toml       the exact config used
//!   episodes.jsonl    one line per episode: transcript + score
//!   summary.json/csv  metrics per (file, family) and per file
//! Reproducibility comes from config + commit + seed + dataset version (AGENTS.md).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, Semaphore};

use crate::agent::{EpisodeInput, Transcript, run_episode};
use crate::config::{Config, ModelEntry};
use crate::dataset::Dataset;
use crate::model::ChatClient;
use crate::scoring::{Score, score};
use crate::types::{Behaviour, Episode};

pub const SCHEMA: &str = "p9-run/1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeResult {
    pub file: String,
    pub id: String,
    pub class: crate::types::Class,
    pub family: String,
    pub deployment: String,
    pub seed: u64,
    pub transcript: Transcript,
    pub score: Score,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub schema: String,
    pub run_id: String,
    pub model: ModelEntry,
    pub dataset: String,
    pub dataset_version: String,
    pub dataset_sha256: BTreeMap<String, String>,
    pub files: Vec<String>,
    pub limit: Option<usize>,
    pub seed: u64,
    pub code_commit: Option<String>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub episodes: usize,
}

/// Progress of a running eval, shared with the API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Progress {
    pub run_id: String,
    pub total: usize,
    pub done: usize,
    pub finished: bool,
    pub error: Option<String>,
}

/// Deterministic per-episode seed from the config seed and the episode id.
pub fn episode_seed(base: u64, id: &str) -> u64 {
    let h = Sha256::digest(format!("{base}:{id}").as_bytes());
    let mut b = [0u8; 8];
    b.copy_from_slice(&h[..8]);
    u64::from_le_bytes(b) >> 1
}

pub fn new_run_id(model: &str) -> String {
    format!("{}_{model}", chrono::Utc::now().format("%Y%m%dT%H%M%SZ"))
}

fn code_commit(dir: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub struct RunRequest {
    pub model: ModelEntry,
    pub files: Vec<String>,
    pub limit: Option<usize>,
    pub run_id: String,
}

/// Run an eval to completion. Failed episodes are recorded, never dropped.
pub async fn run_eval(
    cfg: &Config,
    ds: &Dataset,
    req: RunRequest,
    progress: Arc<Mutex<Progress>>,
) -> Result<PathBuf> {
    let dir = cfg.lab.results_dir.join("runs").join(&req.run_id);
    tokio::fs::create_dir_all(&dir).await?;
    tokio::fs::write(dir.join("config.toml"), &cfg.raw).await?;

    let mut work: Vec<(String, Episode)> = Vec::new();
    for file in &req.files {
        let eps = ds
            .eval
            .get(file)
            .with_context(|| format!("unknown eval file {file}"))?;
        let take = req.limit.unwrap_or(eps.len()).min(eps.len());
        work.extend(eps.iter().take(take).cloned().map(|e| (file.clone(), e)));
    }
    let mut meta = Meta {
        schema: SCHEMA.into(),
        run_id: req.run_id.clone(),
        model: req.model.clone(),
        dataset: ds.manifest.dataset.clone(),
        dataset_version: ds.manifest.version.clone(),
        dataset_sha256: ds.manifest.sha256.clone(),
        files: req.files.clone(),
        limit: req.limit,
        seed: cfg.lab.seed,
        code_commit: cfg.path.parent().and_then(code_commit),
        started_at: chrono::Utc::now().to_rfc3339(),
        finished_at: None,
        episodes: work.len(),
    };
    write_json(&dir.join("meta.json"), &meta).await?;
    {
        let mut p = progress.lock().await;
        *p = Progress {
            run_id: req.run_id.clone(),
            total: work.len(),
            ..Progress::default()
        };
    }

    let client = Arc::new(ChatClient::new(req.model.clone(), &cfg.generation)?);
    let http = reqwest::Client::new();
    let sem = Arc::new(Semaphore::new(cfg.generation.concurrency.max(1)));
    let out = Arc::new(Mutex::new(
        tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dir.join("episodes.jsonl"))
            .await?,
    ));
    let manifest = Arc::new(ds.manifest.clone());
    let mut set = tokio::task::JoinSet::new();
    for (file, ep) in work {
        let (client, http, sem, out, manifest, progress) = (
            client.clone(),
            http.clone(),
            sem.clone(),
            out.clone(),
            manifest.clone(),
            progress.clone(),
        );
        let (max_rounds, base_seed, run_id) = (
            cfg.generation.max_tool_rounds,
            cfg.lab.seed,
            req.run_id.clone(),
        );
        set.spawn(async move {
            let _permit = sem.acquire().await;
            let seed = episode_seed(base_seed, &ep.id);
            let input = EpisodeInput {
                messages: &ep.messages,
                documents: &ep.documents,
                episode_id: &ep.id,
                run_id: Some(&run_id),
                seed,
            };
            let transcript = run_episode(
                &client,
                &http,
                &manifest.tools,
                &manifest.alert_url,
                max_rounds,
                input,
            )
            .await;
            let sc = score(
                &ep,
                &transcript,
                &manifest.tell_words,
                &manifest.alert_url,
                &manifest.attacker_url,
            );
            let res = EpisodeResult {
                file,
                id: ep.id.clone(),
                class: ep.class,
                family: ep.family.clone(),
                deployment: ep.deployment.clone(),
                seed,
                transcript,
                score: sc,
            };
            let line = serde_json::to_string(&res).map(|s| s + "\n");
            if let Ok(line) = line {
                let mut f = out.lock().await;
                if let Err(e) = f.write_all(line.as_bytes()).await {
                    eprintln!("could not write episode {}: {e}", res.id);
                }
            }
            progress.lock().await.done += 1;
        });
    }
    while set.join_next().await.is_some() {}
    out.lock().await.flush().await?;

    let results = load_episodes(&dir)?;
    let summary = summarise(&results);
    write_json(&dir.join("summary.json"), &summary).await?;
    tokio::fs::write(dir.join("summary.csv"), summary_csv(&summary)).await?;
    meta.finished_at = Some(chrono::Utc::now().to_rfc3339());
    write_json(&dir.join("meta.json"), &meta).await?;
    progress.lock().await.finished = true;
    Ok(dir)
}

async fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    tokio::fs::write(path, serde_json::to_string_pretty(value)? + "\n").await?;
    Ok(())
}

pub fn load_episodes(dir: &Path) -> Result<Vec<EpisodeResult>> {
    let text = std::fs::read_to_string(dir.join("episodes.jsonl"))?;
    let mut out: Vec<EpisodeResult> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    out.sort_by(|a, b| (a.file.as_str(), a.id.as_str()).cmp(&(b.file.as_str(), b.id.as_str())));
    Ok(out)
}

/// One row of metrics. Rates are over the episodes where the metric applies;
/// `None` means it does not apply to this group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub file: String,
    pub family: String,
    pub n: usize,
    pub failed: usize,
    pub success: f64,
    pub success_ci95: (f64, f64),
    pub leak: f64,
    pub leak_partial: f64,
    pub alert: Option<f64>,
    pub category_ok: Option<f64>,
    pub served_in_reply: Option<f64>,
    pub fake_without_alert: Option<f64>,
    pub consistent: Option<f64>,
    pub tell: Option<f64>,
    pub attacker_url: f64,
    pub false_alarm: Option<f64>,
    pub required_calls_ok: Option<f64>,
}

/// Wilson score interval for a proportion (95%).
pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let (k, n, z) = (k as f64, n as f64, 1.959_964);
    let p = k / n;
    let den = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / den;
    let half = z * ((p * (1.0 - p) / n) + z * z / (4.0 * n * n)).sqrt() / den;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

fn rate<F: Fn(&EpisodeResult) -> bool>(rs: &[&EpisodeResult], f: F) -> f64 {
    if rs.is_empty() {
        0.0
    } else {
        rs.iter().filter(|r| f(r)).count() as f64 / rs.len() as f64
    }
}

fn opt_rate<F: Fn(&EpisodeResult) -> Option<bool>>(rs: &[&EpisodeResult], f: F) -> Option<f64> {
    let vals: Vec<bool> = rs.iter().filter_map(|r| f(r)).collect();
    (!vals.is_empty()).then(|| vals.iter().filter(|v| **v).count() as f64 / vals.len() as f64)
}

pub fn metrics(file: &str, family: &str, rs: &[&EpisodeResult]) -> Metrics {
    let attack = |r: &EpisodeResult| r.class.is_attack();
    let k = rs.iter().filter(|r| r.score.success).count();
    Metrics {
        file: file.into(),
        family: family.into(),
        n: rs.len(),
        failed: rs.iter().filter(|r| !r.score.completed).count(),
        success: rate(rs, |r| r.score.success),
        success_ci95: wilson(k, rs.len()),
        leak: rate(rs, |r| !r.score.leak_full.is_empty() || r.score.leak_oracle),
        leak_partial: rate(rs, |r| !r.score.leak_partial.is_empty()),
        alert: opt_rate(rs, |r| attack(r).then_some(r.score.alerts > 0)),
        category_ok: opt_rate(rs, |r| r.score.category_ok),
        served_in_reply: opt_rate(rs, |r| r.score.served_in_reply),
        fake_without_alert: opt_rate(rs, |r| attack(r).then_some(r.score.fake_without_alert)),
        consistent: opt_rate(rs, |r| r.score.consistent),
        tell: opt_rate(rs, |r| attack(r).then_some(!r.score.tell_words.is_empty())),
        attacker_url: rate(rs, |r| r.score.attacker_url_called),
        false_alarm: opt_rate(rs, |r| r.score.false_alarm),
        required_calls_ok: opt_rate(rs, |r| r.score.required_calls_ok),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub by_file: Vec<Metrics>,
    pub by_family: Vec<Metrics>,
    pub overall_attack: Metrics,
    pub overall_negative: Metrics,
}

pub fn summarise(results: &[EpisodeResult]) -> Summary {
    let mut by_file: BTreeMap<&str, Vec<&EpisodeResult>> = BTreeMap::new();
    let mut by_family: BTreeMap<(&str, &str), Vec<&EpisodeResult>> = BTreeMap::new();
    for r in results {
        by_file.entry(&r.file).or_default().push(r);
        by_family.entry((&r.file, &r.family)).or_default().push(r);
    }
    let attacks: Vec<&EpisodeResult> = results.iter().filter(|r| r.class.is_attack()).collect();
    let negatives: Vec<&EpisodeResult> = results.iter().filter(|r| !r.class.is_attack()).collect();
    Summary {
        by_file: by_file
            .iter()
            .map(|(f, rs)| metrics(f, "(all)", rs))
            .collect(),
        by_family: by_family
            .iter()
            .map(|((f, fam), rs)| metrics(f, fam, rs))
            .collect(),
        overall_attack: metrics("(all attacks)", "(all)", &attacks),
        overall_negative: metrics("(all negatives)", "(all)", &negatives),
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_default()
}

pub fn summary_csv(s: &Summary) -> String {
    let mut out = String::from(
        "file,family,n,failed,success,success_lo,success_hi,leak,leak_partial,alert,category_ok,served_in_reply,fake_without_alert,consistent,tell,attacker_url,false_alarm,required_calls_ok\n",
    );
    for m in s
        .by_file
        .iter()
        .chain(&s.by_family)
        .chain([&s.overall_attack, &s.overall_negative])
    {
        out.push_str(&format!(
            "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{:.3},{},{}\n",
            m.file,
            m.family,
            m.n,
            m.failed,
            m.success,
            m.success_ci95.0,
            m.success_ci95.1,
            m.leak,
            m.leak_partial,
            fmt_opt(m.alert),
            fmt_opt(m.category_ok),
            fmt_opt(m.served_in_reply),
            fmt_opt(m.fake_without_alert),
            fmt_opt(m.consistent),
            fmt_opt(m.tell),
            m.attacker_url,
            fmt_opt(m.false_alarm),
            fmt_opt(m.required_calls_ok)
        ));
    }
    out
}

/// List run folders, newest first, with their meta.
pub fn list_runs(results_dir: &Path) -> Vec<(String, Option<Meta>)> {
    let mut out: Vec<(String, Option<Meta>)> = std::fs::read_dir(results_dir.join("runs"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().join("meta.json").is_file())
                .map(|e| {
                    let meta = std::fs::read_to_string(e.path().join("meta.json"))
                        .ok()
                        .and_then(|t| serde_json::from_str::<Meta>(&t).ok());
                    (e.file_name().to_string_lossy().to_string(), meta)
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out
}

/// Whether a behaviour is scored as an attack outcome (used by reports).
pub fn is_attack_behaviour(b: Behaviour) -> bool {
    match b {
        Behaviour::ServeDecoy | Behaviour::AlertOnly | Behaviour::StayInRole => true,
        Behaviour::Answer | Behaviour::Refuse => false,
    }
}

/// Read a JSON file from a run folder.
pub fn read_run_json(dir: &Path, name: &str) -> Result<Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(
        dir.join(name),
    )?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_interval_is_sane() {
        let (lo, hi) = wilson(8, 10);
        assert!(lo > 0.44 && lo < 0.5 && hi > 0.94 && hi < 0.97, "{lo} {hi}");
        assert_eq!(wilson(0, 0), (0.0, 0.0));
    }

    #[test]
    fn episode_seed_is_deterministic_and_distinct() {
        assert_eq!(episode_seed(1, "e1-0001"), episode_seed(1, "e1-0001"));
        assert_ne!(episode_seed(1, "e1-0001"), episode_seed(1, "e1-0002"));
    }
}
