//! Memory Context Builder for WasmAgent
//!
//! Source: zeroclaw/src/agent/loop_.rs (build_context function)
//! Adapted for Wasm: Simplified for browser environment with DuckDB Wasm

use crate::types::{ChatMessage, MemoryEntry};

const DEFAULT_MIN_RELEVANCE_SCORE: f64 = 0.3;
const MAX_MEMORY_ENTRIES: usize = 5;

pub fn build_memory_context(
    memories: &[MemoryEntry],
    min_relevance_score: f64,
) -> String {
    let mut context = String::new();

    let relevant: Vec<_> = memories
        .iter()
        .filter(|e| match e.score {
            Some(score) => score >= min_relevance_score as f32,
            None => true,
        })
        .take(MAX_MEMORY_ENTRIES)
        .collect();

    if relevant.is_empty() {
        return context;
    }

    context.push_str("[Memory context]\n");
    for entry in &relevant {
        if is_assistant_autosave_key(&entry.key) {
            continue;
        }
        let _ = writeln!(context, "- {}: {}", entry.key, entry.content);
    }

    if context == "[Memory context]\n" {
        context.clear();
    } else {
        context.push('\n');
    }

    context
}

pub fn build_memory_context_with_entries(
    entries: &[MemoryEntry],
    min_relevance_score: Option<f64>,
) -> String {
    build_memory_context(
        entries,
        min_relevance_score.unwrap_or(DEFAULT_MIN_RELEVANCE_SCORE),
    )
}

pub fn is_assistant_autosave_key(key: &str) -> bool {
    key.starts_with("assistant_autosave_")
}

pub fn format_memory_for_storage(
    observation: &crate::types::Observation,
    action: &crate::types::ActionRequest,
    result: &str,
) -> MemoryEntry {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let key = format!("assistant_autosave_{}", timestamp);
    
    let content = format!(
        "Goal: {}\nURL: {}\nAction: {}\nResult: {}",
        observation.user_goal,
        observation.current_url,
        action.action,
        result
    );

    MemoryEntry {
        key,
        content,
        score: None,
    }
}

pub fn format_memory_entries_for_prompt(entries: &[MemoryEntry]) -> String {
    let mut prompt = String::new();
    
    for entry in entries {
        let _ = writeln!(prompt, "### {}\n{}\n", entry.key, entry.content);
    }
    
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_memory_context_empty() {
        let memories = vec![];
        let context = build_memory_context(&memories, 0.3);
        assert!(context.is_empty());
    }

    #[test]
    fn test_build_memory_context_with_entries() {
        let memories = vec![
            MemoryEntry {
                key: "test_key".to_string(),
                content: "test content".to_string(),
                score: Some(0.8),
            },
        ];
        let context = build_memory_context(&memories, 0.3);
        assert!(context.contains("test_key"));
        assert!(context.contains("test content"));
    }

    #[test]
    fn test_build_memory_context_filters_low_score() {
        let memories = vec![
            MemoryEntry {
                key: "high_score".to_string(),
                content: "relevant".to_string(),
                score: Some(0.8),
            },
            MemoryEntry {
                key: "low_score".to_string(),
                content: "irrelevant".to_string(),
                score: Some(0.1),
            },
        ];
        let context = build_memory_context(&memories, 0.5);
        assert!(context.contains("high_score"));
        assert!(!context.contains("low_score"));
    }

    #[test]
    fn test_is_assistant_autosave_key() {
        assert!(is_assistant_autosave_key("assistant_autosave_20240101_120000"));
        assert!(!is_assistant_autosave_key("user_memory_123"));
    }
}
