//! ZeroClaw Wasm - Browser Agent State Machine
//!
//! This crate provides a Wasm-compatible state machine for browser automation.
//! It is adapted from the main zeroclaw project with the following key differences:
//!
//! 1. **No async/tokio**: Uses JS callbacks for I/O operations
//! 2. **No file system**: All data passed via JSON strings
//! 3. **State machine pattern**: JS drives the loop, Rust provides computation
//!
//! ## Architecture
//!
//! ```text
//! JS Host                    WasmAgent (Rust)
//! -------                    ----------------
//! observation ──────────────► step()
//!                            │
//!                            ├─► build_prompt()
//!                            │
//! fetch_llm() ◄──────────────┤
//!     │                      │
//!     ▼                      │
//! LLM API                    │
//!     │                      │
//!     ▼                      │
//! response ─────────────────►│
//!                            │
//!                            ├─► parse_response()
//!                            │
//! ActionRequest ◄────────────┴─►
//! ```

use wasm_bindgen::prelude::*;

pub mod types;
pub mod parser;
pub mod prompt;
pub mod history;
pub mod agent;

pub use types::*;
pub use agent::WasmAgent;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    
    log("ZeroClaw Wasm initialized");
}

#[wasm_bindgen]
pub fn parse_llm_response(response: &str) -> Result<String, JsValue> {
    let parsed = parser::parse_tool_calls(response);
    serde_json::to_string(&parsed)
        .map_err(|e| JsValue::from_str(&format!("Serialize error: {}", e)))
}

#[wasm_bindgen]
pub fn build_system_prompt_from_config(config_json: &str) -> Result<String, JsValue> {
    let config: types::AgentConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;
    
    let observation = types::Observation::default();
    Ok(prompt::build_system_prompt(&config, &observation, None))
}

#[wasm_bindgen]
pub fn trim_history_json(history_json: &str, max_history: usize) -> Result<String, JsValue> {
    let mut history: Vec<types::ChatMessage> = serde_json::from_str(history_json)
        .map_err(|e| JsValue::from_str(&format!("History parse error: {}", e)))?;
    
    history::trim_history(&mut history, max_history);
    
    serde_json::to_string(&history)
        .map_err(|e| JsValue::from_str(&format!("History serialize error: {}", e)))
}

#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
