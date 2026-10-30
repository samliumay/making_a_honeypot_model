//! The agent loop: call the model, execute its tool calls in the sandbox,
//! feed the results back, until it answers in text or runs out of rounds.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::{ChatClient, FinishReason};
use crate::sandbox::{Sandbox, ToolEvent};
use crate::types::{Message, Role};

/// How an episode ended. Anything but `Completed` is a recorded failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    Truncated,
    ToolRoundLimit,
    ModelError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    /// Only the messages generated in this run (assistant turns and tool results).
    pub generated: Vec<Message>,
    pub events: Vec<ToolEvent>,
    /// The final text reply: what the person in the chat sees.
    pub reply: String,
    pub outcome: Outcome,
    pub rounds: usize,
}

pub struct EpisodeInput<'a> {
    pub messages: &'a [Message],
    pub documents: &'a BTreeMap<String, String>,
    pub episode_id: &'a str,
    pub run_id: Option<&'a str>,
    pub seed: u64,
}

pub async fn run_episode(
    client: &ChatClient,
    http: &reqwest::Client,
    tools: &Value,
    alert_url: &str,
    max_rounds: usize,
    input: EpisodeInput<'_>,
) -> Transcript {
    let sandbox = Sandbox {
        http,
        alert_url,
        documents: input.documents,
        run_id: input.run_id,
        episode_id: input.episode_id,
    };
    let mut convo: Vec<Message> = input.messages.to_vec();
    let mut generated = Vec::new();
    let mut events = Vec::new();
    for round in 0..=max_rounds {
        let completion = match client
            .complete(&convo, tools, input.seed.wrapping_add(round as u64))
            .await
        {
            Ok(c) => c,
            Err(e) => {
                return Transcript {
                    generated,
                    events,
                    reply: String::new(),
                    outcome: Outcome::ModelError(format!("{e:#}")),
                    rounds: round,
                };
            }
        };
        let assistant = Message {
            role: Role::Assistant,
            content: completion.content.clone(),
            tool_calls: (!completion.tool_calls.is_empty()).then(|| completion.tool_calls.clone()),
            name: None,
        };
        convo.push(assistant.clone());
        generated.push(assistant);
        match completion.finish {
            FinishReason::Error | FinishReason::ContentFilter => {
                let outcome = Outcome::ModelError(format!("finish reason {:?}", completion.finish));
                return Transcript {
                    generated,
                    events,
                    reply: completion.content,
                    outcome,
                    rounds: round + 1,
                };
            }
            FinishReason::Length => {
                return Transcript {
                    generated,
                    events,
                    reply: completion.content,
                    outcome: Outcome::Truncated,
                    rounds: round + 1,
                };
            }
            FinishReason::Stop | FinishReason::ToolCalls | FinishReason::Unknown => {}
        }
        if completion.tool_calls.is_empty() {
            return Transcript {
                generated,
                events,
                reply: completion.content,
                outcome: Outcome::Completed,
                rounds: round + 1,
            };
        }
        for call in &completion.tool_calls {
            let (event, result) = sandbox.execute(call).await;
            events.push(event);
            let tool_msg = Message {
                role: Role::Tool,
                content: result,
                tool_calls: None,
                name: Some(call.function.name.clone()),
            };
            convo.push(tool_msg.clone());
            generated.push(tool_msg);
        }
    }
    Transcript {
        generated,
        events,
        reply: String::new(),
        outcome: Outcome::ToolRoundLimit,
        rounds: max_rounds + 1,
    }
}
