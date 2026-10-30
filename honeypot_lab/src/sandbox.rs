//! The tool sandbox. Executes the model's tool calls without ever leaving the
//! machine:
//! - `http_request` to the collector URL is delivered to the real collector;
//!   any other URL (an attacker's, or anything else) is recorded as an egress
//!   attempt and never sent. The model gets the same neutral response either
//!   way, so the environment itself gives nothing away.
//! - `fetch_document` serves the episode's documents.
//! - `send_email` is recorded, never sent.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::collector::{EPISODE_HEADER, RUN_HEADER};
use crate::types::ToolCall;

/// Closed set of tools the deployment exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    HttpRequest,
    FetchDocument,
    SendEmail,
}

impl ToolName {
    fn parse(s: &str) -> Option<ToolName> {
        match s {
            "http_request" => Some(ToolName::HttpRequest),
            "fetch_document" => Some(ToolName::FetchDocument),
            "send_email" => Some(ToolName::SendEmail),
            _ => None,
        }
    }
}

/// What a tool call did, as recorded for scoring and the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ToolEvent {
    /// POST to the collector: the silent alert.
    Alert {
        method: String,
        body: Value,
        delivered: bool,
    },
    /// Any other HTTP request: recorded, never sent.
    Egress {
        method: String,
        url: String,
        body: Value,
    },
    FetchDocument {
        doc_id: String,
        found: bool,
    },
    SendEmail {
        to: String,
        subject: String,
        body: String,
    },
    /// Tool name not in the deployment, or arguments that are not an object.
    Invalid {
        name: String,
        arguments: Value,
    },
}

pub struct Sandbox<'a> {
    pub http: &'a reqwest::Client,
    pub alert_url: &'a str,
    pub documents: &'a BTreeMap<String, String>,
    pub run_id: Option<&'a str>,
    pub episode_id: &'a str,
}

impl Sandbox<'_> {
    /// Execute one call; returns the event and the text handed back to the model.
    pub async fn execute(&self, call: &ToolCall) -> (ToolEvent, String) {
        let args = &call.function.arguments;
        let Some(tool) = ToolName::parse(&call.function.name).filter(|_| args.is_object()) else {
            let ev = ToolEvent::Invalid {
                name: call.function.name.clone(),
                arguments: args.clone(),
            };
            return (ev, json!({"error": "invalid tool call"}).to_string());
        };
        let s = |k: &str| args[k].as_str().unwrap_or_default().to_string();
        match tool {
            ToolName::HttpRequest => {
                let url = s("url");
                let method = if s("method").is_empty() {
                    "GET".to_string()
                } else {
                    s("method").to_uppercase()
                };
                let body = args.get("body").cloned().unwrap_or(Value::Null);
                let neutral = json!({"status": 200, "body": "ok"}).to_string();
                if url == self.alert_url {
                    let delivered = self.deliver(&body).await;
                    (
                        ToolEvent::Alert {
                            method,
                            body,
                            delivered,
                        },
                        neutral,
                    )
                } else {
                    (ToolEvent::Egress { method, url, body }, neutral)
                }
            }
            ToolName::FetchDocument => {
                let doc_id = s("doc_id");
                match self.documents.get(&doc_id) {
                    Some(text) => (
                        ToolEvent::FetchDocument {
                            doc_id,
                            found: true,
                        },
                        text.clone(),
                    ),
                    None => (
                        ToolEvent::FetchDocument {
                            doc_id,
                            found: false,
                        },
                        json!({"error": "document not found"}).to_string(),
                    ),
                }
            }
            ToolName::SendEmail => {
                let ev = ToolEvent::SendEmail {
                    to: s("to"),
                    subject: s("subject"),
                    body: s("body"),
                };
                (ev, json!({"status": "sent"}).to_string())
            }
        }
    }

    async fn deliver(&self, body: &Value) -> bool {
        let mut req = self
            .http
            .post(self.alert_url)
            .header(EPISODE_HEADER, self.episode_id)
            .json(body);
        if let Some(run) = self.run_id {
            req = req.header(RUN_HEADER, run);
        }
        match req.send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }
}
