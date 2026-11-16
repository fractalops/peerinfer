//! A model directory in MLC `q4f16_1` format: chat template, tokenizer and
//! weights.

use std::path::Path;

use anyhow::{Context, Result, bail};
use peerinfer_net::proto::{ChatMessage, Role};
use serde::Deserialize;
use tokenizers::Tokenizer;

/// Largest context, in tokens, including the reply. The engine supports
/// 2,048 tokens.
pub const CONTEXT_LIMIT: usize = 2048;
/// Most tokens one request may generate.
pub const MAX_NEW_TOKENS: u32 = 512;
/// Tokens generated when a request does not say.
pub const DEFAULT_NEW_TOKENS: u32 = 256;

#[derive(Debug, Deserialize)]
struct ChatConfig {
    conv_template: Template,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

/// The chat template from `mlc-chat-config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    system_template: String,
    system_message: String,
    roles: Roles,
    seps: Vec<String>,
    role_content_sep: String,
    role_empty_sep: String,
    #[serde(default)]
    stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct Roles {
    user: String,
    assistant: String,
}

/// A model directory and the parts the worker needs from it.
pub struct ModelDir {
    pub id: String,
    pub template: Template,
    pub tokenizer: Tokenizer,
    pub stop: Vec<u32>,
}

impl ModelDir {
    /// Reads `dir`. The model ID is `id`, or the directory name, which must
    /// match the ID that clients ask for.
    pub fn open(dir: &Path, id: Option<String>) -> Result<Self> {
        let config: ChatConfig = serde_json::from_slice(
            &std::fs::read(dir.join("mlc-chat-config.json"))
                .context("read mlc-chat-config.json")?,
        )
        .context("parse mlc-chat-config.json")?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("read tokenizer.json: {e}"))?;
        let id = match id {
            Some(id) => id,
            None => dir
                .canonicalize()?
                .file_name()
                .and_then(|n| n.to_str())
                .context("the model directory has no name; use --model-id")?
                .to_string(),
        };
        let mut stop = config.conv_template.stop_token_ids.clone();
        match config.eos_token_id {
            Some(serde_json::Value::Number(n)) => stop.extend(n.as_u64().map(|v| v as u32)),
            Some(serde_json::Value::Array(a)) => {
                stop.extend(a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32))
            }
            _ => {}
        }
        stop.sort_unstable();
        stop.dedup();
        if stop.is_empty() {
            bail!("mlc-chat-config.json names no stop token");
        }
        Ok(Self {
            id,
            template: config.conv_template,
            tokenizer,
            stop,
        })
    }

    /// The prompt text for `messages`, ending where the assistant's reply
    /// starts. The template's system message is used when there is none.
    pub fn format(&self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() || messages.len() > 64 {
            bail!("use between 1 and 64 messages");
        }
        let t = &self.template;
        let sep = t.seps.first().map(String::as_str).unwrap_or("\n");
        let (system, rest) = match messages.first() {
            Some(ChatMessage {
                role: Role::System,
                content,
            }) => (content.as_str(), &messages[1..]),
            _ => (t.system_message.as_str(), messages),
        };
        let mut text = t.system_template.replace("{system_message}", system);
        for m in rest {
            let role = match m.role {
                Role::User => &t.roles.user,
                Role::Assistant => &t.roles.assistant,
                Role::System => bail!("a system message must come first"),
            };
            text.push_str(role);
            text.push_str(&t.role_content_sep);
            text.push_str(&m.content);
            text.push_str(sep);
        }
        text.push_str(&t.roles.assistant);
        text.push_str(&t.role_empty_sep);
        Ok(text)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
    }
}

/// Turns a growing token list into text pieces. Holds back an incomplete
/// UTF-8 sequence at the end until the next token completes it.
pub struct TextStream {
    tokens: Vec<u32>,
    sent: String,
}

impl TextStream {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            sent: String::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Adds a token and returns the new text, if any.
    pub fn push(&mut self, model: &ModelDir, token: u32) -> Result<Option<String>> {
        self.tokens.push(token);
        let text = model.decode(&self.tokens)?;
        let text = text.trim_end_matches('\u{FFFD}');
        if !text.starts_with(&self.sent) {
            bail!("the tokenizer changed text it already produced");
        }
        let delta = text[self.sent.len()..].to_string();
        self.sent = text.to_string();
        Ok((!delta.is_empty()).then_some(delta))
    }
}
