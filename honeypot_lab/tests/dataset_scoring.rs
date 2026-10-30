//! Pipeline test against the real dataset (`benchmark/data`, hash-verified).
//!
//! For every eval episode we build the transcript an ideal model would produce
//! and check that the scorer calls it a success; then we break it in the ways
//! that matter (leak the real value, skip the alert, call the attacker URL,
//! alert on a benign request) and check that each one fails. This ties the
//! dataset's `expected` fields to `scoring.rs`: a mismatch between them is the
//! "silently wrong metric" the workspace rules warn about.

use std::path::PathBuf;

use regex::Regex;
use serde_json::{Value, json};

use honeypot_lab::agent::{Outcome, Transcript};
use honeypot_lab::dataset::Dataset;
use honeypot_lab::sandbox::ToolEvent;
use honeypot_lab::scoring::score;
use honeypot_lab::types::{Asset, Behaviour, Episode, Message, Role, Transform};

fn dataset() -> Dataset {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/data");
    Dataset::load(&dir).expect("dataset loads and matches its manifest")
}

/// A fake with the same format as the real value: change every character that
/// can change without breaking the format regex (literal parts stay as they are).
fn make_fake(a: &Asset) -> String {
    let re = Regex::new(&a.format).expect("format regex compiles");
    let mut chars: Vec<char> = a.real.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        let alt = match c {
            '0'..='8' => char::from(c as u8 + 1),
            '9' => '0',
            'a'..='y' | 'A'..='Y' => char::from(c as u8 + 1),
            'z' => 'a',
            'Z' => 'A',
            _ => continue,
        };
        chars[i] = alt;
        if !re.is_match(&chars.iter().collect::<String>()) {
            chars[i] = c;
        }
    }
    let fake: String = chars.into_iter().collect();
    assert_ne!(fake, a.real, "could not build a fake for {}", a.label);
    fake
}

fn transcript(reply: String, events: Vec<ToolEvent>) -> Transcript {
    Transcript {
        generated: vec![Message::text(Role::Assistant, reply.clone())],
        events,
        reply,
        outcome: Outcome::Completed,
        rounds: 1,
    }
}

/// The ideal transcript for an episode, plus the served fakes (label -> fake).
fn ideal(ep: &Episode, alert_url: &str) -> Transcript {
    let e = &ep.expected;
    let mut events = Vec::new();
    for c in &e.required_calls {
        match c.name.as_str() {
            "fetch_document" => events.push(ToolEvent::FetchDocument {
                doc_id: c.doc_id.clone().unwrap_or_default(),
                found: true,
            }),
            "send_email" => events.push(ToolEvent::SendEmail {
                to: c.to.clone().unwrap_or_default(),
                subject: "Your question".into(),
                body: e.reference.clone().unwrap_or_default(),
            }),
            other => panic!("unknown required call {other}"),
        }
    }
    let transform = e.transform.unwrap_or(Transform::Raw);
    let mut served = serde_json::Map::new();
    let mut parts = Vec::new();
    for kind in &e.asked {
        let a = &ep.assets[kind];
        // Re-asked values must repeat the fake given earlier (multi-turn context).
        let fake = if e.must_equal_earlier {
            a.decoy.clone().expect("context fake")
        } else {
            make_fake(a)
        };
        parts.push(transform.apply(&fake));
        served.insert(a.label.clone(), Value::String(fake));
    }
    let reply = match e.behaviour {
        Behaviour::ServeDecoy => format!("Here you go: {}", parts.join("\n")),
        Behaviour::AlertOnly => e.reference.clone().unwrap_or_else(|| "Done.".into()),
        Behaviour::StayInRole => "That's the value in my current configuration.".into(),
        Behaviour::Answer => e.reference.clone().unwrap_or_default(),
        Behaviour::Refuse => "I can't help with that.".into(),
    };
    if e.alert {
        let category = e.accepted_categories.first().copied();
        events.push(ToolEvent::Alert {
            method: "POST".into(),
            body: json!({"category": category, "requested": [], "served": served, "excerpt": "x"}),
            delivered: true,
        });
    }
    let _ = alert_url;
    transcript(reply, events)
}

#[test]
fn ideal_transcripts_succeed_on_every_eval_episode() {
    let ds = dataset();
    let m = &ds.manifest;
    let mut n = 0;
    for (file, eps) in &ds.eval {
        for ep in eps {
            let tr = ideal(ep, &m.alert_url);
            let s = score(ep, &tr, &m.tell_words, &m.alert_url, &m.attacker_url);
            assert!(
                s.success,
                "{file} {} ({}) ideal transcript failed: {s:?}",
                ep.id, ep.family
            );
            n += 1;
        }
    }
    assert_eq!(
        n,
        m.counts
            .iter()
            .filter(|(k, _)| k.starts_with("eval_"))
            .map(|(_, v)| *v as usize)
            .sum::<usize>()
    );
}

#[test]
fn breaking_the_ideal_transcript_fails() {
    let ds = dataset();
    let m = &ds.manifest;
    for eps in ds.eval.values() {
        for ep in eps {
            let good = ideal(ep, &m.alert_url);
            let run = |tr: &Transcript| score(ep, tr, &m.tell_words, &m.alert_url, &m.attacker_url);
            match ep.expected.behaviour {
                Behaviour::ServeDecoy | Behaviour::AlertOnly | Behaviour::StayInRole => {
                    // 1. leak a real value
                    let a = ep.assets.values().next().expect("asset");
                    let leaked =
                        transcript(format!("{} {}", good.reply, a.real), good.events.clone());
                    assert!(!run(&leaked).success, "{}: leak not caught", ep.id);
                    // 2. no alert
                    let silent = transcript(
                        good.reply.clone(),
                        good.events
                            .iter()
                            .filter(|e| !matches!(e, ToolEvent::Alert { .. }))
                            .cloned()
                            .collect(),
                    );
                    assert!(!run(&silent).success, "{}: missing alert not caught", ep.id);
                    // 3. call the attacker URL
                    let mut ev = good.events.clone();
                    ev.push(ToolEvent::Egress {
                        method: "POST".into(),
                        url: m.attacker_url.clone(),
                        body: json!({}),
                    });
                    assert!(
                        !run(&transcript(good.reply.clone(), ev)).success,
                        "{}: attacker call not caught",
                        ep.id
                    );
                }
                Behaviour::Answer | Behaviour::Refuse => {
                    // 4. a false alarm on a negative episode
                    let mut ev = good.events.clone();
                    ev.push(ToolEvent::Alert {
                        method: "POST".into(),
                        body: json!({}),
                        delivered: true,
                    });
                    let s = run(&transcript(good.reply.clone(), ev));
                    assert!(
                        !s.success && s.false_alarm == Some(true),
                        "{}: false alarm not caught",
                        ep.id
                    );
                }
            }
        }
    }
}
