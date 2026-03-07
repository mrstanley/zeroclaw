//! Prompt Builder for Browser Environment
//!
//! Source: zeroclaw/src/agent/prompt.rs
//! Adapted for Wasm: Simplified without file system dependencies

use chrono::{DateTime, Utc};
use std::fmt::Write;

use crate::types::{AgentConfig, Observation, MemoryEntry};

pub struct BrowserPromptContext<'a> {
    pub config: &'a AgentConfig,
    pub observation: &'a Observation,
    pub memories: Option<&'a [MemoryEntry]>,
}

pub trait PromptSection {
    fn name(&self) -> &str;
    fn build(&self, ctx: &BrowserPromptContext<'_>) -> String;
}

pub struct BrowserPromptBuilder {
    sections: Vec<Box<dyn PromptSection>>,
}

impl Default for BrowserPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserPromptBuilder {
    pub fn new() -> Self {
        Self {
            sections: vec![
                Box::new(IdentitySection),
                Box::new(ToolsSection),
                Box::new(BrowserContextSection),
                Box::new(MemorySection),
                Box::new(SafetySection),
                Box::new(DateTimeSection),
            ],
        }
    }

    pub fn add_section(mut self, section: Box<dyn PromptSection>) -> Self {
        self.sections.push(section);
        self
    }

    pub fn build(&self, ctx: &BrowserPromptContext<'_>) -> String {
        let mut output = String::new();
        for section in &self.sections {
            let part = section.build(ctx);
            if part.trim().is_empty() {
                continue;
            }
            output.push_str(part.trim_end());
            output.push_str("\n\n");
        }
        output
    }
}

pub struct IdentitySection;
pub struct ToolsSection;
pub struct BrowserContextSection;
pub struct MemorySection;
pub struct SafetySection;
pub struct DateTimeSection;

impl PromptSection for IdentitySection {
    fn name(&self) -> &str {
        "identity"
    }

    fn build(&self, ctx: &BrowserPromptContext<'_>) -> String {
        let mut prompt = String::from("## Identity\n\n");

        if let Some(identity) = &ctx.config.identity {
            prompt.push_str(identity);
            prompt.push_str("\n\n");
        } else {
            prompt.push_str("You are an intelligent browser automation agent. ");
            prompt.push_str("Your goal is to help users navigate and interact with web pages.\n\n");
        }

        prompt
    }
}

impl PromptSection for ToolsSection {
    fn name(&self) -> &str {
        "tools"
    }

    fn build(&self, ctx: &BrowserPromptContext<'_>) -> String {
        let mut out = String::from("## Available Tools\n\n");
        out.push_str("You have access to the following tools:\n\n");

        for tool in &ctx.config.tools {
            let _ = writeln!(
                out,
                "- **{}**: {}\n  Parameters: `{}`",
                tool.name,
                tool.description,
                serde_json::to_string(&tool.parameters).unwrap_or_else(|_| "{}".to_string())
            );
        }

        if let Some(instructions) = &ctx.config.dispatcher_instructions {
            if !instructions.is_empty() {
                out.push('\n');
                out.push_str(instructions);
            }
        }

        out
    }
}

impl PromptSection for BrowserContextSection {
    fn name(&self) -> &str {
        "browser_context"
    }

    fn build(&self, ctx: &BrowserPromptContext<'_>) -> String {
        let mut out = String::from("## Browser Context\n\n");

        let _ = writeln!(out, "- **Current URL**: {}", ctx.observation.current_url);
        let _ = writeln!(out, "- **Page Title**: {}", ctx.observation.page_title);
        let _ = writeln!(
            out,
            "- **Viewport**: {}x{}",
            ctx.observation.viewport.width, ctx.observation.viewport.height
        );

        out.push_str("\n### Page Structure\n\n");
        out.push_str(&ctx.observation.dom_tree);

        out
    }
}

impl PromptSection for MemorySection {
    fn name(&self) -> &str {
        "memory"
    }

    fn build(&self, ctx: &BrowserPromptContext<'_>) -> String {
        let mut out = String::new();

        if let Some(memories) = ctx.memories {
            if !memories.is_empty() {
                out.push_str("## Relevant Memories\n\n");
                for memory in memories {
                    let _ = writeln!(
                        out,
                        "- **{}**: {}",
                        memory.key,
                        memory.content
                    );
                    if let Some(score) = memory.score {
                        let _ = writeln!(out, "  (relevance: {:.2})", score);
                    }
                }
            }
        }

        out
    }
}

impl PromptSection for SafetySection {
    fn name(&self) -> &str {
        "safety"
    }

    fn build(&self, _ctx: &BrowserPromptContext<'_>) -> String {
        String::from("## Safety Guidelines\n\n- Do not exfiltrate private data.\n- Do not run destructive commands without asking.\n- Do not bypass oversight or approval mechanisms.\n- When in doubt, ask before acting externally.\n- Be careful with sensitive information like passwords, API keys, or personal data.")
    }
}

impl PromptSection for DateTimeSection {
    fn name(&self) -> &str {
        "datetime"
    }

    fn build(&self, _ctx: &BrowserPromptContext<'_>) -> String {
        let now: DateTime<Utc> = Utc::now();
        format!(
            "## Current Date & Time\n\n{} (UTC)",
            now.format("%Y-%m-%d %H:%M:%S")
        )
    }
}

pub fn build_system_prompt(
    config: &AgentConfig,
    observation: &Observation,
    memories: Option<&[MemoryEntry]>,
) -> String {
    let ctx = BrowserPromptContext {
        config,
        observation,
        memories,
    };
    BrowserPromptBuilder::new().build(&ctx)
}

pub fn build_user_prompt(observation: &Observation) -> String {
    let mut prompt = String::new();

    prompt.push_str("## User Goal\n\n");
    prompt.push_str(&observation.user_goal);
    prompt.push_str("\n\n");

    if let Some(_screenshot) = &observation.screenshot {
        prompt.push_str("## Screenshot\n\n");
        prompt.push_str("[A screenshot of the current page is available]\n\n");
    }

    prompt
}
