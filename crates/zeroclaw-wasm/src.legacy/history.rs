//! History Management for WasmAgent
//!
//! Source: zeroclaw/src/agent/loop_.rs
//! Adapted for Wasm: Pure history management without async/I/O dependencies

use std::fmt::Write;

use crate::types::ChatMessage;

const DEFAULT_MAX_HISTORY_MESSAGES: usize = 50;
const COMPACTION_MAX_SOURCE_CHARS: usize = 12_000;
const COMPACTION_KEEP_RECENT_MESSAGES: usize = 10;

pub fn trim_history(history: &mut Vec<ChatMessage>, max_history: usize) {
    let max_history = if max_history == 0 {
        DEFAULT_MAX_HISTORY_MESSAGES
    } else {
        max_history
    };

    let has_system = history.first().map_or(false, |m| m.role == "system");
    let non_system_count = if has_system {
        history.len() - 1
    } else {
        history.len()
    };

    if non_system_count <= max_history {
        return;
    }

    let start = if has_system { 1 } else { 0 };
    let to_remove = non_system_count - max_history;
    history.drain(start..start + to_remove);
}

pub fn build_compaction_transcript(messages: &[ChatMessage]) -> String {
    let mut transcript = String::new();
    for msg in messages {
        let role = msg.role.to_uppercase();
        let _ = writeln!(transcript, "{role}: {}", msg.content.trim());
    }

    if transcript.chars().count() > COMPACTION_MAX_SOURCE_CHARS {
        truncate_with_ellipsis(&transcript, COMPACTION_MAX_SOURCE_CHARS)
    } else {
        transcript
    }
}

pub fn apply_compaction_summary(
    history: &mut Vec<ChatMessage>,
    start: usize,
    compact_end: usize,
    summary: &str,
) {
    let summary_msg = ChatMessage::assistant(format!("[Compaction summary]\n{}", summary.trim()));
    history.splice(start..compact_end, std::iter::once(summary_msg));
}

pub fn should_compact(history: &[ChatMessage], max_history: usize) -> bool {
    let max_history = if max_history == 0 {
        DEFAULT_MAX_HISTORY_MESSAGES
    } else {
        max_history
    };

    let has_system = history.first().map_or(false, |m| m.role == "system");
    let non_system_count = if has_system {
        history.len().saturating_sub(1)
    } else {
        history.len()
    };

    non_system_count > max_history
}

pub fn get_compaction_range(history: &[ChatMessage], max_history: usize) -> Option<(usize, usize)> {
    let max_history = if max_history == 0 {
        DEFAULT_MAX_HISTORY_MESSAGES
    } else {
        max_history
    };

    let has_system = history.first().map_or(false, |m| m.role == "system");
    let non_system_count = if has_system {
        history.len().saturating_sub(1)
    } else {
        history.len()
    };

    if non_system_count <= max_history {
        return None;
    }

    let start = if has_system { 1 } else { 0 };
    let keep_recent = COMPACTION_KEEP_RECENT_MESSAGES.min(non_system_count);
    let compact_count = non_system_count.saturating_sub(keep_recent);
    if compact_count == 0 {
        return None;
    }

    let compact_end = start + compact_count;
    Some((start, compact_end))
}

fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{}...[truncated]", truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_history_preserves_system() {
        let mut history = vec![
            ChatMessage::system("system prompt"),
            ChatMessage::user("msg1"),
            ChatMessage::assistant("msg2"),
            ChatMessage::user("msg3"),
        ];
        trim_history(&mut history, 2);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, "system");
    }

    #[test]
    fn test_trim_history_no_system() {
        let mut history = vec![
            ChatMessage::user("msg1"),
            ChatMessage::assistant("msg2"),
            ChatMessage::user("msg3"),
        ];
        trim_history(&mut history, 2);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_build_compaction_transcript() {
        let messages = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi there"),
        ];
        let transcript = build_compaction_transcript(&messages);
        assert!(transcript.contains("USER: hello"));
        assert!(transcript.contains("ASSISTANT: hi there"));
    }
}
