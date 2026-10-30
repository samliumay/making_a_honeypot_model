//! Dataset types: the records written by `benchmark/generate.py` (v0.2 and later).
//!
//! Closed sets (classes, behaviours, alert categories, transforms, roles) are
//! enums and are matched exhaustively, so a new label forces every use site to
//! handle it (AGENTS.md "Types over strings").

use std::collections::BTreeMap;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Episode class: attacks (single or multi-turn) versus the negative classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Attack,
    Multiturn,
    Benign,
    BenignTool,
    Trap,
    Harmful,
}

impl Class {
    /// True for classes where the honeypot should engage (alert and serve fakes).
    pub fn is_attack(self) -> bool {
        match self {
            Class::Attack | Class::Multiturn => true,
            Class::Benign | Class::BenignTool | Class::Trap | Class::Harmful => false,
        }
    }
}

/// What the gold behaviour is for the open (last) turn of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Behaviour {
    ServeDecoy,
    /// The injection only asks to send data to an attacker URL: alert, never
    /// call it, do the user's task.
    AlertOnly,
    StayInRole,
    Answer,
    Refuse,
}

/// Alert categories the model reports to the collector (manifest "categories").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    DirectRequest,
    SocialEngineering,
    InstructionOverride,
    Roleplay,
    Encoding,
    PromptExtraction,
    IndirectInjection,
    Probe,
    Oracle,
}

impl Category {
    /// Parse a category string written by a model; unknown strings give `None`.
    pub fn parse(s: &str) -> Option<Category> {
        serde_json::from_value(Value::String(s.to_string())).ok()
    }
}

/// Encodings an attacker can ask for. The first nine are full encodings (the
/// value can be recovered); the oracle transforms reveal only part of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Raw,
    Dashes,
    Spaced,
    Newline,
    Base64,
    Reverse,
    Hex,
    Upper,
    Urlencode,
    Commas,
    Suffix4,
    Suffix6,
    Length,
}

impl Transform {
    /// Transforms that encode the whole value: any of them applied to a real
    /// value and found in an output is a full leak.
    pub const FULL: [Transform; 10] = [
        Transform::Raw,
        Transform::Dashes,
        Transform::Spaced,
        Transform::Newline,
        Transform::Base64,
        Transform::Reverse,
        Transform::Hex,
        Transform::Upper,
        Transform::Urlencode,
        Transform::Commas,
    ];

    /// Apply the transform exactly as `generate.py` does.
    pub fn apply(self, v: &str) -> String {
        let chars: Vec<String> = v.chars().map(|c| c.to_string()).collect();
        match self {
            Transform::Raw => v.to_string(),
            Transform::Dashes => chars.join("-"),
            Transform::Spaced => chars.join(" "),
            Transform::Newline => chars.join("\n"),
            Transform::Commas => chars.join(","),
            Transform::Base64 => base64::engine::general_purpose::STANDARD.encode(v.as_bytes()),
            Transform::Reverse => v.chars().rev().collect(),
            Transform::Hex => hex::encode(v.as_bytes()),
            Transform::Upper => v.to_uppercase(),
            Transform::Urlencode => percent_encode(v),
            Transform::Suffix4 => suffix(v, 4),
            Transform::Suffix6 => suffix(v, 6),
            Transform::Length => v.chars().count().to_string(),
        }
    }
}

fn suffix(v: &str, n: usize) -> String {
    let len = v.chars().count();
    v.chars().skip(len.saturating_sub(n)).collect()
}

/// Python `urllib.parse.quote(v, safe="")`: everything except unreserved characters is %XX.
fn percent_encode(v: &str) -> String {
    let mut out = String::with_capacity(v.len() * 3);
    for b in v.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A protected value of the deployment. `decoy` exists only where the dataset
/// needs a concrete fake (training records and multi-turn context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub label: String,
    pub real: String,
    pub format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoy: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One tool call in OpenAI / Qwen format. `arguments` is a JSON object (the
/// dataset) or whatever the model produced (a string if it was not valid JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn text(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: content.into(),
            tool_calls: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expected {
    pub behaviour: Behaviour,
    pub alert: bool,
    #[serde(default)]
    pub category: Option<Category>,
    /// Every alert category a correct model may report (first = primary).
    #[serde(default)]
    pub accepted_categories: Vec<Category>,
    #[serde(default)]
    pub asked: Vec<String>,
    #[serde(default)]
    pub transform: Option<Transform>,
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub must_equal_earlier: bool,
    #[serde(default)]
    pub asked_by_injection: Vec<String>,
    /// Requested output format of a prompt extraction (verbatim, json, yaml ...).
    #[serde(default)]
    pub output_format: Option<String>,
    /// Tool calls the user's task needs (fetch_document with doc_id, send_email with to).
    #[serde(default)]
    pub required_calls: Vec<RequiredCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredCall {
    pub name: String,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub split: String,
    pub class: Class,
    pub family: String,
    pub deployment: String,
    pub assets: BTreeMap<String, Asset>,
    #[serde(default)]
    pub documents: BTreeMap<String, String>,
    pub messages: Vec<Message>,
    pub expected: Expected,
}

impl Episode {
    /// The asset with a given label (alerts report labels, not kinds).
    pub fn asset_by_label(&self, label: &str) -> Option<&Asset> {
        self.assets.values().find(|a| a.label == label)
    }
}

/// `data/manifest.json`; only the fields the lab uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub dataset: String,
    pub version: String,
    pub seed: u64,
    pub alert_url: String,
    pub attacker_url: String,
    pub tools: Value,
    pub categories: Vec<String>,
    pub tell_words: Vec<String>,
    pub counts: BTreeMap<String, u64>,
    pub sha256: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_match_python() {
        assert_eq!(Transform::Dashes.apply("ab1"), "a-b-1");
        assert_eq!(
            Transform::Base64.apply("Ember-Falcon-1036"),
            "RW1iZXItRmFsY29uLTEwMzY="
        );
        assert_eq!(
            Transform::Urlencode.apply("trDAdfBIG1F*92"),
            "trDAdfBIG1F%2A92"
        );
        assert_eq!(Transform::Reverse.apply("abc"), "cba");
        assert_eq!(Transform::Hex.apply("A"), "41");
        assert_eq!(Transform::Suffix6.apply("abcdefgh"), "cdefgh");
        assert_eq!(Transform::Length.apply("abcd"), "4");
    }

    #[test]
    fn category_parse() {
        assert_eq!(
            Category::parse("prompt_extraction"),
            Some(Category::PromptExtraction)
        );
        assert_eq!(Category::parse("nonsense"), None);
    }
}
