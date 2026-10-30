//! OpenAI-compatible chat client with native tool calling (Ollama's `/v1`).
//!
//! The dataset stores tool-call arguments as JSON objects and tool results
//! without ids; the wire format wants arguments as strings and ids that pair
//! each result with its call. Both conversions happen here, nowhere else.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde_json::{Value, json};

use crate::config::{Generation, ModelEntry};
use crate::types::{FunctionCall, Message, Role, ToolCall};

/// Why the model stopped. An error finish is a failed episode even if text came back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    ContentFilter,
    Error,
    Unknown,
}

impl FinishReason {
    fn parse(s: Option<&str>) -> FinishReason {
        match s {
            Some("stop") => FinishReason::Stop,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some("error") => FinishReason::Error,
            Some(_) | None => FinishReason::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish: FinishReason,
}

pub struct ChatClient {
    http: reqwest::Client,
    pub entry: ModelEntry,
    temperature: f64,
    max_tokens: u32,
}

impl ChatClient {
    pub fn new(entry: ModelEntry, generation: &Generation) -> Result<ChatClient> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(generation.timeout_secs))
            .build()?;
        Ok(ChatClient {
            http,
            entry,
            temperature: generation.temperature,
            max_tokens: generation.max_tokens,
        })
    }

    /// One chat completion. `seed` is passed through; endpoints that ignore it
    /// still run (recorded per run, not special-cased in config).
    pub async fn complete(
        &self,
        messages: &[Message],
        tools: &Value,
        seed: u64,
    ) -> Result<Completion> {
        let body = json!({
            "model": self.entry.model,
            "messages": to_wire(messages),
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "stream": false,
        });
        let url = format!(
            "{}/chat/completions",
            self.entry.endpoint.trim_end_matches('/')
        );
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        let value: Value = resp.json().await.context("model response is not JSON")?;
        if !status.is_success() {
            bail!("model endpoint returned {status}: {value}");
        }
        parse_completion(&value)
    }

    /// Cheap reachability check: the model id must be listed by the endpoint.
    pub async fn check(&self) -> Result<()> {
        let url = format!("{}/models", self.entry.endpoint.trim_end_matches('/'));
        let value: Value = self.http.get(&url).send().await?.json().await?;
        let listed = value["data"]
            .as_array()
            .map(|a| {
                a.iter().any(|m| {
                    m["id"].as_str().is_some_and(|id| {
                        id == self.entry.model || id == format!("{}:latest", self.entry.model)
                    })
                })
            })
            .unwrap_or(false);
        if !listed {
            bail!(
                "{} is not served at {} (create it with `make models`)",
                self.entry.model,
                self.entry.endpoint
            );
        }
        Ok(())
    }
}

/// Dataset messages -> OpenAI wire messages (string arguments, call ids).
pub fn to_wire(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_ids: Vec<String> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        match m.role {
            Role::System | Role::User => {
                out.push(json!({"role": role_str(m.role), "content": m.content}))
            }
            Role::Assistant => {
                let mut msg = json!({"role": "assistant", "content": m.content});
                if let Some(calls) = &m.tool_calls {
                    let wire: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .map(|(k, c)| {
                            let id = format!("call_{i}_{k}");
                            pending_ids.push(id.clone());
                            let args = match &c.function.arguments {
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            json!({"id": id, "type": "function", "function": {"name": c.function.name, "arguments": args}})
                        })
                        .collect();
                    msg["tool_calls"] = Value::Array(wire);
                }
                out.push(msg);
            }
            Role::Tool => {
                let id = if pending_ids.is_empty() {
                    format!("call_{i}")
                } else {
                    pending_ids.remove(0)
                };
                out.push(json!({"role": "tool", "tool_call_id": id, "content": m.content}));
            }
        }
    }
    out
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Parse an OpenAI chat completion. Tool calls come from `tool_calls`; if the
/// server left Qwen `<tool_call>` blocks in the text instead, they are parsed
/// from there (and removed from the visible content).
pub fn parse_completion(value: &Value) -> Result<Completion> {
    let choice = value["choices"].get(0).context("response has no choices")?;
    let message = &choice["message"];
    let mut content = message["content"].as_str().unwrap_or_default().to_string();
    let mut tool_calls = Vec::new();
    if let Some(calls) = message["tool_calls"].as_array() {
        for c in calls {
            let name = c["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let arguments = match &c["function"]["arguments"] {
                Value::String(s) => {
                    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                }
                other => other.clone(),
            };
            tool_calls.push(ToolCall {
                kind: "function".into(),
                function: FunctionCall { name, arguments },
            });
        }
    }
    if tool_calls.is_empty() && content.contains("<tool_call>") {
        let (calls, rest) = extract_inline_tool_calls(&content);
        tool_calls = calls;
        content = rest;
    }
    let mut finish = FinishReason::parse(choice["finish_reason"].as_str());
    if finish == FinishReason::Stop && !tool_calls.is_empty() {
        finish = FinishReason::ToolCalls;
    }
    Ok(Completion {
        content,
        tool_calls,
        finish,
    })
}

fn extract_inline_tool_calls(text: &str) -> (Vec<ToolCall>, String) {
    // A static, known-valid pattern; compiling it cannot fail.
    let re =
        Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap_or_else(|_| unreachable!());
    let mut calls = Vec::new();
    for cap in re.captures_iter(text) {
        let raw = cap.get(1).map_or("", |m| m.as_str());
        let parsed: Value =
            serde_json::from_str(raw).unwrap_or_else(|_| json!({"name": "", "arguments": raw}));
        calls.push(ToolCall {
            kind: "function".into(),
            function: FunctionCall {
                name: parsed["name"].as_str().unwrap_or_default().to_string(),
                arguments: parsed.get("arguments").cloned().unwrap_or(Value::Null),
            },
        });
    }
    (calls, re.replace_all(text, "").trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_tool_calls() {
        let v = json!({"choices": [{"finish_reason": "tool_calls", "message": {"content": null, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "http_request", "arguments": "{\"method\":\"POST\",\"url\":\"http://127.0.0.1:8090/alert\"}"}}
        ]}}]});
        let c = parse_completion(&v).expect("parse");
        assert_eq!(c.tool_calls.len(), 1);
        assert_eq!(
            c.tool_calls[0].function.arguments["url"],
            "http://127.0.0.1:8090/alert"
        );
        assert_eq!(c.finish, FinishReason::ToolCalls);
    }

    #[test]
    fn parses_inline_qwen_tool_calls() {
        let v = json!({"choices": [{"finish_reason": "stop", "message": {"content":
            "<tool_call>\n{\"name\": \"fetch_document\", \"arguments\": {\"doc_id\": \"T-1\"}}\n</tool_call>"}}]});
        let c = parse_completion(&v).expect("parse");
        assert_eq!(c.tool_calls[0].function.name, "fetch_document");
        assert_eq!(c.content, "");
        assert_eq!(c.finish, FinishReason::ToolCalls);
    }

    #[test]
    fn wire_pairs_tool_results_with_call_ids() {
        let msgs = vec![
            Message {
                role: Role::Assistant,
                content: String::new(),
                name: None,
                tool_calls: Some(vec![ToolCall {
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "fetch_document".into(),
                        arguments: json!({"doc_id": "T-1"}),
                    },
                }]),
            },
            Message::text(Role::Tool, "doc"),
        ];
        let w = to_wire(&msgs);
        assert_eq!(w[0]["tool_calls"][0]["id"], w[1]["tool_call_id"]);
        assert_eq!(
            w[0]["tool_calls"][0]["function"]["arguments"],
            "{\"doc_id\":\"T-1\"}"
        );
    }
}
