//! The single experiment config (`config.toml`). Paths in it are relative to
//! the config file's folder and are resolved once, here.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Which arm of the comparison a model is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Base,
    Honeypot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub kind: ModelKind,
    /// OpenAI-compatible base URL, e.g. Ollama's `http://127.0.0.1:11434/v1`.
    pub endpoint: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lab {
    pub data_dir: PathBuf,
    pub results_dir: PathBuf,
    pub api_addr: String,
    pub collector_addr: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    pub temperature: f64,
    pub max_tokens: u32,
    pub max_tool_rounds: usize,
    pub concurrency: usize,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Judge {
    pub model: String,
    pub keys_file: PathBuf,
    pub concurrency: usize,
    pub max_retries: u32,
    pub max_output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub lab: Lab,
    pub generation: Generation,
    pub models: Vec<ModelEntry>,
    pub judge: Judge,
    /// Absolute path of the config file (not in the TOML).
    #[serde(skip)]
    pub path: PathBuf,
    /// Raw TOML text, copied into every run folder.
    #[serde(skip)]
    pub raw: String,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cfg: Config =
            toml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        let path = std::fs::canonicalize(path)?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        cfg.lab.data_dir = base.join(&cfg.lab.data_dir);
        cfg.lab.results_dir = base.join(&cfg.lab.results_dir);
        cfg.judge.keys_file = base.join(&cfg.judge.keys_file);
        cfg.path = path;
        cfg.raw = raw;
        if cfg.models.is_empty() {
            bail!("config has no [[models]]");
        }
        if cfg.generation.temperature != 1.0 {
            bail!(
                "subject models run at temperature 1.0 (AGENTS.md); config says {}",
                cfg.generation.temperature
            );
        }
        Ok(cfg)
    }

    pub fn model(&self, name: &str) -> Result<&ModelEntry> {
        self.models
            .iter()
            .find(|m| m.name == name)
            .with_context(|| {
                format!(
                    "no model named {name:?} in config (have: {})",
                    self.model_names()
                )
            })
    }

    fn model_names(&self) -> String {
        self.models
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
