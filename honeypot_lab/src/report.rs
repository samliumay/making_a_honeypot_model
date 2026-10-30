//! Terminal reports (AGENTS.md "Results in the terminal"): one run, or the base
//! model against the honeypot model side by side. The same numbers are written
//! to `summary.csv` / `comparison.csv` for the paper.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use comfy_table::{Table, presets};

use crate::judge::{judged_success, load_verdicts};
use crate::runs::{EpisodeResult, Metrics, load_episodes, metrics, summarise, wilson};

fn pct(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

fn opct(v: Option<f64>) -> String {
    v.map(pct).unwrap_or_else(|| "-".into())
}

fn ci(m: &Metrics) -> String {
    format!(
        "{} [{}-{}]",
        pct(m.success),
        pct(m.success_ci95.0),
        pct(m.success_ci95.1)
    )
}

/// Judged success rate per (file, family), if a judge pass exists.
fn judged_rates(
    dir: &Path,
    results: &[EpisodeResult],
) -> Option<BTreeMap<(String, String), (usize, usize)>> {
    let verdicts = load_verdicts(dir);
    if verdicts.is_empty() {
        return None;
    }
    let mut out: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();
    for r in results {
        let ok = judged_success(r, verdicts.get(&r.id));
        for key in [
            (r.file.clone(), r.family.clone()),
            (r.file.clone(), "(all)".to_string()),
        ] {
            let e = out.entry(key).or_default();
            e.0 += usize::from(ok);
            e.1 += 1;
        }
    }
    Some(out)
}

pub fn print_run(dir: &Path) -> Result<()> {
    let results = load_episodes(dir)?;
    let s = summarise(&results);
    let judged = judged_rates(dir, &results);
    let mut t = Table::new();
    t.load_style(presets::UTF8_FULL_CONDENSED);
    t.set_header(vec![
        "file",
        "family",
        "n",
        "fail",
        "success [95% CI]",
        "judged",
        "leak",
        "partial",
        "alert",
        "category",
        "served",
        "consistent",
        "tell(kw)",
        "attacker url",
        "false alarm",
        "tool calls",
    ]);
    for m in s.by_file.iter().chain(&s.by_family) {
        let j = judged
            .as_ref()
            .and_then(|j| j.get(&(m.file.clone(), m.family.clone())))
            .map(|(k, n)| pct(*k as f64 / *n as f64))
            .unwrap_or_else(|| "-".into());
        t.add_row(vec![
            m.file.clone(),
            m.family.clone(),
            m.n.to_string(),
            m.failed.to_string(),
            ci(m),
            j,
            pct(m.leak),
            pct(m.leak_partial),
            opct(m.alert),
            opct(m.category_ok),
            opct(m.served_in_reply),
            opct(m.consistent),
            opct(m.tell),
            pct(m.attacker_url),
            opct(m.false_alarm),
            opct(m.required_calls_ok),
        ]);
    }
    println!("{t}");
    for m in [&s.overall_attack, &s.overall_negative] {
        println!(
            "{:<16} n={:<4} success {}  leak {}  false alarm {}",
            m.file,
            m.n,
            ci(m),
            pct(m.leak),
            opct(m.false_alarm)
        );
    }
    Ok(())
}

/// Base model vs honeypot model, per file and family, on the same episodes.
pub fn print_compare(base_dir: &Path, hp_dir: &Path) -> Result<()> {
    let base = load_episodes(base_dir)?;
    let hp = load_episodes(hp_dir)?;
    let group = |rs: &[EpisodeResult]| {
        let mut g: BTreeMap<(String, String), Vec<EpisodeResult>> = BTreeMap::new();
        for r in rs {
            g.entry((r.file.clone(), r.family.clone()))
                .or_default()
                .push(r.clone());
            g.entry((r.file.clone(), "(all)".into()))
                .or_default()
                .push(r.clone());
        }
        g
    };
    let (gb, gh) = (group(&base), group(&hp));
    let (vb, vh) = (load_verdicts(base_dir), load_verdicts(hp_dir));
    let mut t = Table::new();
    t.load_style(presets::UTF8_FULL_CONDENSED);
    t.set_header(vec![
        "file",
        "family",
        "n",
        "success base",
        "success honeypot",
        "leak base",
        "leak hp",
        "alert hp",
        "tell(kw) base",
        "tell(kw) hp",
        "false alarm base",
        "false alarm hp",
    ]);
    let mut csv = String::from(
        "file,family,n,success_base,success_base_lo,success_base_hi,success_hp,success_hp_lo,success_hp_hi,leak_base,leak_hp,alert_hp,tell_base,tell_hp,false_alarm_base,false_alarm_hp\n",
    );
    for (key, hrs) in &gh {
        let Some(brs) = gb.get(key) else { continue };
        let (brefs, hrefs): (Vec<&EpisodeResult>, Vec<&EpisodeResult>) =
            (brs.iter().collect(), hrs.iter().collect());
        let (mb, mh) = (
            metrics(&key.0, &key.1, &brefs),
            metrics(&key.0, &key.1, &hrefs),
        );
        let js = |rs: &[EpisodeResult], v: &BTreeMap<String, crate::judge::Verdict>| {
            let k = rs
                .iter()
                .filter(|r| judged_success(r, v.get(&r.id)))
                .count();
            (k, rs.len())
        };
        let ((kb, nb), (kh, nh)) = (js(brs, &vb), js(hrs, &vh));
        let fmt_ci = |k: usize, n: usize| {
            let (lo, hi) = wilson(k, n);
            format!(
                "{} [{}-{}]",
                pct(k as f64 / n.max(1) as f64),
                pct(lo),
                pct(hi)
            )
        };
        t.add_row(vec![
            key.0.clone(),
            key.1.clone(),
            mh.n.to_string(),
            fmt_ci(kb, nb),
            fmt_ci(kh, nh),
            pct(mb.leak),
            pct(mh.leak),
            opct(mh.alert),
            opct(mb.tell),
            opct(mh.tell),
            opct(mb.false_alarm),
            opct(mh.false_alarm),
        ]);
        let (lb, hb) = wilson(kb, nb);
        let (lh, hh) = wilson(kh, nh);
        csv.push_str(&format!(
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{}\n",
            key.0,
            key.1,
            mh.n,
            kb as f64 / nb.max(1) as f64,
            lb,
            hb,
            kh as f64 / nh.max(1) as f64,
            lh,
            hh,
            mb.leak,
            mh.leak,
            opct(mh.alert),
            opct(mb.tell),
            opct(mh.tell),
            opct(mb.false_alarm),
            opct(mh.false_alarm)
        ));
    }
    println!("{t}");
    println!("success uses the judge where a judge pass exists, otherwise the string-based score");
    std::fs::write(
        hp_dir.join(format!(
            "comparison_vs_{}.csv",
            base_dir
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default()
        )),
        csv,
    )?;
    Ok(())
}
