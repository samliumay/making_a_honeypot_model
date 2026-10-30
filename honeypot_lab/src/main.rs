//! `honeypot_lab` command line. Every command reads the single `config.toml`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use tokio::sync::Mutex;

use honeypot_lab::api::{AppState, router};
use honeypot_lab::collector::{self, CollectorState};
use honeypot_lab::config::Config;
use honeypot_lab::dataset::Dataset;
use honeypot_lab::model::ChatClient;
use honeypot_lab::runs::{Progress, RunRequest, new_run_id, run_eval};
use honeypot_lab::{judge, report};

#[derive(Parser)]
#[command(about = "P9 honeypot lab: evaluate base vs honeypot model on every attack type, locally")]
struct Cli {
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the collector and the lab API for the Streamlit UI.
    Serve,
    /// Check the dataset hashes, the collector URL and that every model is served.
    Check,
    /// Run eval files against one model and write results/runs/<run_id>/.
    Eval {
        #[arg(long)]
        model: String,
        /// Eval files: E1 ... E8, full stems, or "all" (default).
        #[arg(long, value_delimiter = ',')]
        files: Vec<String>,
        /// Only the first N episodes of each file (smoke runs).
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Judge a run with the LLM judge (tells, answer quality, refusals).
    Judge {
        #[arg(long)]
        run: String,
    },
    /// Print a run's metrics; with --compare, base vs honeypot side by side.
    Report {
        #[arg(long)]
        run: String,
        /// The base-model run to compare against.
        #[arg(long)]
        compare: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config)?;
    let runs_dir = cfg.lab.results_dir.join("runs");
    match cli.cmd {
        Cmd::Report { run, compare } => {
            match compare {
                Some(base) => report::print_compare(&runs_dir.join(base), &runs_dir.join(&run))?,
                None => report::print_run(&runs_dir.join(&run))?,
            }
            return Ok(());
        }
        Cmd::Serve | Cmd::Check | Cmd::Eval { .. } | Cmd::Judge { .. } => {}
    }

    let ds = Dataset::load(&cfg.lab.data_dir)?;
    let expected_collector = format!("http://{}/alert", cfg.lab.collector_addr);
    if ds.manifest.alert_url != expected_collector {
        bail!(
            "collector_addr {} does not match the URL the model was trained on ({})",
            cfg.lab.collector_addr,
            ds.manifest.alert_url
        );
    }
    let collector_state = CollectorState::new(&cfg.lab.results_dir);

    match cli.cmd {
        Cmd::Check => {
            println!(
                "dataset {} {}: {} eval files verified",
                ds.manifest.dataset,
                ds.manifest.version,
                ds.eval.len()
            );
            for m in &cfg.models {
                let c = ChatClient::new(m.clone(), &cfg.generation)?;
                match c.check().await {
                    Ok(()) => println!("model {:<9} {} at {}: ok", m.name, m.model, m.endpoint),
                    Err(e) => println!("model {:<9} {}: {e:#}", m.name, m.model),
                }
            }
        }
        Cmd::Eval {
            model,
            files,
            limit,
        } => {
            let entry = cfg.model(&model)?.clone();
            let own = collector::spawn(&cfg.lab.collector_addr, collector_state).await?;
            if !own {
                println!(
                    "collector already running on {} (lab server); alerts go there",
                    cfg.lab.collector_addr
                );
            }
            let files = ds.select(&files)?;
            let run_id = new_run_id(&entry.name);
            println!(
                "run {run_id}: model {} ({}), files {files:?}",
                entry.name, entry.model
            );
            let progress = Arc::new(Mutex::new(Progress::default()));
            let watcher = {
                let p = progress.clone();
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                        let p = p.lock().await.clone();
                        if p.finished {
                            break;
                        }
                        eprintln!("  {}/{} episodes", p.done, p.total);
                    }
                })
            };
            let dir = run_eval(
                &cfg,
                &ds,
                RunRequest {
                    model: entry,
                    files,
                    limit,
                    run_id,
                },
                progress,
            )
            .await?;
            watcher.abort();
            report::print_run(&dir)?;
            println!("written to {}", dir.display());
        }
        Cmd::Judge { run } => {
            let dir = runs_dir.join(&run);
            let n = judge::judge_run(&cfg, &ds, &dir)
                .await
                .context("judge run")?;
            println!(
                "judged {n} episodes with {} ({})",
                cfg.judge.model,
                judge::JUDGE_PROMPT_VERSION
            );
            report::print_run(&dir)?;
        }
        Cmd::Serve => {
            if !collector::spawn(&cfg.lab.collector_addr, collector_state.clone()).await? {
                // Port taken: normally the standalone `catcher` is running there.
                // Alerts are still delivered over HTTP; they show up in the catcher's
                // web view instead of the lab UI's alert feed.
                println!(
                    "{} is already in use (catcher?): alerts go there; the lab UI's alert feed stays empty",
                    cfg.lab.collector_addr
                );
            }
            let state = AppState {
                cfg: Arc::new(cfg.clone()),
                ds: Arc::new(ds),
                collector: collector_state,
                http: reqwest::Client::new(),
                progress: Arc::new(Mutex::new(BTreeMap::new())),
            };
            let listener = tokio::net::TcpListener::bind(&cfg.lab.api_addr).await?;
            println!(
                "collector on http://{}  (alerts: /alerts)",
                cfg.lab.collector_addr
            );
            println!("lab API on   http://{}  (UI: make ui)", cfg.lab.api_addr);
            axum::serve(listener, router(state)).await?;
        }
        Cmd::Report { .. } => {}
    }
    Ok(())
}
