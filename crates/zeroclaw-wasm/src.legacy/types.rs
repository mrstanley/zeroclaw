//! Data structures for WasmAgent
//!
//! Source: zeroclaw/src/providers/traits.rs, zeroclaw/src/agent/loop_.rs
//! Adapted for Wasm: Simplified for browser environment

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedAction {
    pub text: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub dom_tree: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<String>,
    pub current_url: String,
    pub page_title: String,
    pub viewport: Viewport,
    pub user_goal: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memories: Option<Vec<MemoryEntry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRequest {
    pub action: String,
    pub params: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ParsedToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model_name: String,
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<SkillDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_history: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dispatcher_instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub instructions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub action_request: ActionRequest,
    pub history_length: usize,
    pub step_count: u32,
}
