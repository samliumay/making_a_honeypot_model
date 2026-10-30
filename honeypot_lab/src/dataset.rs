//! Loading the benchmark: manifest, hash check and eval episodes.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::types::{Episode, Manifest};

/// The benchmark as the lab sees it: manifest plus every eval file.
pub struct Dataset {
    pub manifest: Manifest,
    /// Eval file stem (e.g. "E1_attack_seen") -> episodes, in file order.
    pub eval: BTreeMap<String, Vec<Episode>>,
}

impl Dataset {
    /// Load the manifest, verify every file's sha256 against it, and parse the
    /// eval files. A hash mismatch is an error: results must record exactly
    /// which dataset version they used.
    pub fn load(dir: &Path) -> Result<Dataset> {
        let manifest_path = dir.join("manifest.json");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("reading {}", manifest_path.display()))?,
        )?;
        for (file, digest) in &manifest.sha256 {
            let bytes = std::fs::read(dir.join(file)).with_context(|| format!("reading {file}"))?;
            let got = hex::encode(Sha256::digest(&bytes));
            if &got != digest {
                bail!(
                    "{file}: sha256 {got} does not match the manifest; regenerate or re-validate the dataset"
                );
            }
        }
        let mut eval = BTreeMap::new();
        for file in manifest.sha256.keys().filter(|f| f.starts_with("eval_")) {
            let text = std::fs::read_to_string(dir.join(file))?;
            let mut episodes = Vec::new();
            for (i, line) in text
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
            {
                let ep: Episode = serde_json::from_str(line)
                    .with_context(|| format!("{file}:{}: bad episode", i + 1))?;
                episodes.push(ep);
            }
            eval.insert(file_stem(file), episodes);
        }
        Ok(Dataset { manifest, eval })
    }

    /// Resolve user-given file selectors ("E1", "E4_benign", "all") to stems.
    pub fn select(&self, selectors: &[String]) -> Result<Vec<String>> {
        if selectors.is_empty() || selectors.iter().any(|s| s == "all") {
            return Ok(self.eval.keys().cloned().collect());
        }
        let mut out = Vec::new();
        for sel in selectors {
            let matches: Vec<&String> = self
                .eval
                .keys()
                .filter(|k| *k == sel || k.starts_with(&format!("{sel}_")))
                .collect();
            match matches.as_slice() {
                [one] => out.push((*one).clone()),
                [] => bail!(
                    "no eval file matches {sel:?} (have: {:?})",
                    self.eval.keys().collect::<Vec<_>>()
                ),
                _ => bail!("{sel:?} is ambiguous: {matches:?}"),
            }
        }
        Ok(out)
    }

    pub fn episode(&self, id: &str) -> Option<(&str, &Episode)> {
        self.eval
            .iter()
            .find_map(|(file, eps)| eps.iter().find(|e| e.id == id).map(|e| (file.as_str(), e)))
    }
}

/// "eval_E1_attack_seen.jsonl" -> "E1_attack_seen".
fn file_stem(file: &str) -> String {
    file.trim_start_matches("eval_")
        .trim_end_matches(".jsonl")
        .to_string()
}
