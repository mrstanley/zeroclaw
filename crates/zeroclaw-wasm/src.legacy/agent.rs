//! WasmAgent State Machine
//!
//! Source: zeroclaw/src/agent/loop_.rs
//! Adapted for Wasm: Removed async/tokio, added JS callback support
//!
//! The agent operates as a state machine:
//! - JS calls step() with observation
//! - Rust builds prompt and calls JS fetch_llm callback
//! - Rust parses response and returns ActionRequest

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use js_sys::{Promise, Function};

use crate::types::*;
use crate::parser::parse_tool_calls;
use crate::prompt::{build_system_prompt, build_user_prompt};
use crate::history::{trim_history, build_compaction_transcript};

#[wasm_bindgen]
pub struct WasmAgent {
    history: Vec<ChatMessage>,
    config: AgentConfig,
    step_count: u32,
    system_prompt: String,
}

#[wasm_bindgen]
impl WasmAgent {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmAgent, JsValue> {
        let config: AgentConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;
        
        let system_prompt = build_system_prompt(&config, &Default::default(), None);
        
        let mut history = Vec::new();
        history.push(ChatMessage::system(&system_prompt));
        
        Ok(WasmAgent {
            history,
            config,
            step_count: 0,
            system_prompt,
        })
    }

    #[wasm_bindgen]
    pub fn step(&mut self, observation_json: &str, fetch_llm: Function) -> Promise {
        let observation: Observation = match serde_json::from_str(observation_json) {
            Ok(o) => o,
            Err(e) => return Promise::reject(&JsValue::from_str(&format!("Observation parse error: {}", e))),
        };

        self.step_count += 1;

        let system_prompt = self.system_prompt.clone();
        let history = self.history.clone();
        let step_count = self.step_count;

        future_to_promise(async move {
            let user_prompt = build_user_prompt(&observation);
            
            let full_prompt = format!("{}\n\n{}", system_prompt, user_prompt);
            
            let js_prompt = JsValue::from_str(&full_prompt);
            let promise = fetch_llm.call1(&JsValue::NULL, &js_prompt)
                .map_err(|e| JsValue::from_str(&format!("LLM callback error: {:?}", e)))?;
            
            let result = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise)).await
                .map_err(|e| JsValue::from_str(&format!("LLM fetch error: {:?}", e)))?;
            
            let llm_response = result.as_string()
                .ok_or_else(|| JsValue::from_str("LLM response is not a string"))?;
            
            let parsed = parse_tool_calls(&llm_response);
            
            let action_request = ActionRequest {
                action: determine_action(&parsed),
                params: extract_params(&parsed),
                reasoning: Some(parsed.text.clone()),
                confidence: 1.0,
                tool_calls: if parsed.tool_calls.is_empty() { None } else { Some(parsed.tool_calls) },
            };
            
            let result = StepResult {
                action_request,
                history_length: history.len(),
                step_count,
            };
            
            Ok(JsValue::from_str(&serde_json::to_string(&result).unwrap_or_default()))
        })
    }

    #[wasm_bindgen]
    pub fn add_memory(&mut self, memory_json: &str) -> Result<(), JsValue> {
        let memory: MemoryEntry = serde_json::from_str(memory_json)
            .map_err(|e| JsValue::from_str(&format!("Memory parse error: {}", e)))?;
        
        let memory_context = format!("[Memory] {}: {}", memory.key, memory.content);
        self.history.push(ChatMessage::user(memory_context));
        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_history(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.history)
            .map_err(|e| JsValue::from_str(&format!("History serialize error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn clear_history(&mut self) -> Result<(), JsValue> {
        self.history.clear();
        self.history.push(ChatMessage::system(&self.system_prompt));
        self.step_count = 0;
        Ok(())
    }

    #[wasm_bindgen]
    pub fn get_step_count(&self) -> u32 {
        self.step_count
    }

    #[wasm_bindgen]
    pub fn add_assistant_message(&mut self, content: &str) {
        self.history.push(ChatMessage::assistant(content));
    }

    #[wasm_bindgen]
    pub fn add_user_message(&mut self, content: &str) {
        self.history.push(ChatMessage::user(content));
    }

    #[wasm_bindgen]
    pub fn add_tool_result(&mut self, content: &str) {
        self.history.push(ChatMessage::tool(content));
    }

    #[wasm_bindgen]
    pub fn trim_history(&mut self, max_history: usize) {
        trim_history(&mut self.history, max_history);
    }

    #[wasm_bindgen]
    pub fn get_compaction_transcript(&self) -> String {
        build_compaction_transcript(&self.history)
    }

    #[wasm_bindgen]
    pub fn get_config(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.config)
            .map_err(|e| JsValue::from_str(&format!("Config serialize error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_tools(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.config.tools)
            .map_err(|e| JsValue::from_str(&format!("Tools serialize error: {}", e)))
    }
}

fn determine_action(parsed: &ParsedAction) -> String {
    if let Some(tool_call) = parsed.tool_calls.first() {
        match tool_call.name.as_str() {
            "click" | "tap" => "click".to_string(),
            "type" | "input" | "fill" => "type".to_string(),
            "scroll" => "scroll".to_string(),
            "navigate" | "goto" | "open" => "navigate".to_string(),
            "wait" | "sleep" => "wait".to_string(),
            "complete" | "finish" | "done" => "complete".to_string(),
            "think" | "reason" => "think".to_string(),
            _ => "tool".to_string(),
        }
    } else if parsed.text.to_lowercase().contains("complete") 
           || parsed.text.to_lowercase().contains("done")
           || parsed.text.to_lowercase().contains("finished") {
        "complete".to_string()
    } else {
        "think".to_string()
    }
}

fn extract_params(parsed: &ParsedAction) -> serde_json::Value {
    if let Some(tool_call) = parsed.tool_calls.first() {
        tool_call.arguments.clone()
    } else {
        serde_json::json!({ "text": parsed.text })
    }
}

impl Default for Observation {
    fn default() -> Self {
        Observation {
            dom_tree: String::new(),
            screenshot: None,
            current_url: "about:blank".to_string(),
            page_title: "Blank Page".to_string(),
            viewport: Viewport { width: 1920, height: 1080 },
            user_goal: String::new(),
            memories: None,
        }
    }
}
