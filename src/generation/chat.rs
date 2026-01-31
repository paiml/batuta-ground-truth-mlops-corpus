//! Chat and Conversation Management
//!
//! Multi-turn conversation handling, message roles, and chat templates.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.generation import ChatMessage, MessageRole, create_chat_config
//! msg = create_chat_message("user", "Hello!")
//! ```

/// Message role in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MessageRole {
    /// System message (instructions)
    System,
    /// User message (default)
    #[default]
    User,
    /// Assistant response
    Assistant,
    /// Tool/function result
    Tool,
}

impl MessageRole {
    /// Get role name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "system" => Some(Self::System),
            "user" | "human" => Some(Self::User),
            "assistant" | "ai" | "bot" => Some(Self::Assistant),
            "tool" | "function" => Some(Self::Tool),
            _ => None,
        }
    }

    /// List all roles
    pub fn list_all() -> Vec<Self> {
        vec![Self::System, Self::User, Self::Assistant, Self::Tool]
    }
}

/// Chat template format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateFormat {
    /// Raw/no template (default)
    #[default]
    Raw,
    /// ChatML format
    ChatML,
    /// Llama 2/3 format
    Llama,
    /// Mistral format
    Mistral,
    /// Gemma format
    Gemma,
}

impl ChatTemplateFormat {
    /// Get format name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::ChatML => "chatml",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "raw" | "none" => Some(Self::Raw),
            "chatml" | "chat_ml" => Some(Self::ChatML),
            "llama" | "llama2" | "llama3" => Some(Self::Llama),
            "mistral" => Some(Self::Mistral),
            "gemma" => Some(Self::Gemma),
            _ => None,
        }
    }

    /// List all formats
    pub fn list_all() -> Vec<Self> {
        vec![Self::Raw, Self::ChatML, Self::Llama, Self::Mistral, Self::Gemma]
    }

    /// Get BOS token
    pub fn bos_token(&self) -> &'static str {
        match self {
            Self::Raw => "",
            Self::ChatML => "",
            Self::Llama => "<s>",
            Self::Mistral => "<s>",
            Self::Gemma => "<bos>",
        }
    }

    /// Get EOS token
    pub fn eos_token(&self) -> &'static str {
        match self {
            Self::Raw => "",
            Self::ChatML => "<|im_end|>",
            Self::Llama => "</s>",
            Self::Mistral => "</s>",
            Self::Gemma => "<eos>",
        }
    }
}

/// A chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Message role
    pub role: MessageRole,
    /// Message content
    pub content: String,
    /// Optional name (for multi-user chats)
    pub name: Option<String>,
    /// Tool call ID (for tool messages)
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create new message
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    /// Create user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create tool message
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        let mut msg = Self::new(MessageRole::Tool, content);
        msg.tool_call_id = Some(tool_call_id.into());
        msg
    }

    /// Set name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Estimate token count
    pub fn token_count(&self) -> usize {
        // Rough estimate: ~4 chars per token
        self.content.len() / 4 + 4 // +4 for role tokens
    }

    /// Format for a given template
    pub fn format(&self, template: ChatTemplateFormat) -> String {
        match template {
            ChatTemplateFormat::Raw => {
                format!("{}: {}", self.role.as_str(), self.content)
            }
            ChatTemplateFormat::ChatML => {
                format!(
                    "<|im_start|>{}\n{}<|im_end|>",
                    self.role.as_str(),
                    self.content
                )
            }
            ChatTemplateFormat::Llama => {
                match self.role {
                    MessageRole::System => {
                        format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", self.content)
                    }
                    MessageRole::User => format!("[INST] {} [/INST]", self.content),
                    MessageRole::Assistant => self.content.clone(),
                    MessageRole::Tool => format!("[TOOL] {} [/TOOL]", self.content),
                }
            }
            ChatTemplateFormat::Mistral => {
                match self.role {
                    MessageRole::User => format!("[INST] {} [/INST]", self.content),
                    MessageRole::Assistant => self.content.clone(),
                    _ => format!("{}: {}", self.role.as_str(), self.content),
                }
            }
            ChatTemplateFormat::Gemma => {
                format!("<start_of_turn>{}\n{}<end_of_turn>", self.role.as_str(), self.content)
            }
        }
    }
}

/// Conversation configuration
#[derive(Debug, Clone)]
pub struct ConversationConfig {
    /// Maximum conversation length in tokens
    pub max_tokens: usize,
    /// Template format
    pub template: ChatTemplateFormat,
    /// Keep system message when truncating
    pub keep_system: bool,
    /// Truncation strategy
    pub truncation: TruncationStrategy,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            template: ChatTemplateFormat::Raw,
            keep_system: true,
            truncation: TruncationStrategy::DropOldest,
        }
    }
}

impl ConversationConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max tokens
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set template
    pub fn template(mut self, t: ChatTemplateFormat) -> Self {
        self.template = t;
        self
    }

    /// Set keep system
    pub fn keep_system(mut self, keep: bool) -> Self {
        self.keep_system = keep;
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation = strategy;
        self
    }
}

/// Truncation strategy for long conversations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    /// Drop oldest messages first (default)
    #[default]
    DropOldest,
    /// Drop newest messages first
    DropNewest,
    /// Summarize old messages
    Summarize,
    /// Error on overflow
    Error,
}

impl TruncationStrategy {
    /// Get strategy name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DropOldest => "drop_oldest",
            Self::DropNewest => "drop_newest",
            Self::Summarize => "summarize",
            Self::Error => "error",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "drop_oldest" | "oldest" => Some(Self::DropOldest),
            "drop_newest" | "newest" => Some(Self::DropNewest),
            "summarize" | "summary" => Some(Self::Summarize),
            "error" | "fail" => Some(Self::Error),
            _ => None,
        }
    }

    /// List all strategies
    pub fn list_all() -> Vec<Self> {
        vec![Self::DropOldest, Self::DropNewest, Self::Summarize, Self::Error]
    }
}

/// A conversation (list of messages)
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// Configuration
    pub config: ConversationConfig,
}

impl Conversation {
    /// Create new conversation
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with config
    pub fn with_config(config: ConversationConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Add message
    pub fn add(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    /// Add system message
    pub fn add_system(&mut self, content: impl Into<String>) {
        self.add(ChatMessage::system(content));
    }

    /// Add user message
    pub fn add_user(&mut self, content: impl Into<String>) {
        self.add(ChatMessage::user(content));
    }

    /// Add assistant message
    pub fn add_assistant(&mut self, content: impl Into<String>) {
        self.add(ChatMessage::assistant(content));
    }

    /// Get total token count
    pub fn token_count(&self) -> usize {
        self.messages.iter().map(|m| m.token_count()).sum()
    }

    /// Get number of messages
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Format entire conversation
    pub fn format(&self) -> String {
        self.messages
            .iter()
            .map(|m| m.format(self.config.template))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Clear conversation
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get last message
    pub fn last(&self) -> Option<&ChatMessage> {
        self.messages.last()
    }

    /// Get system message if present
    pub fn system_message(&self) -> Option<&ChatMessage> {
        self.messages.iter().find(|m| m.role == MessageRole::System)
    }
}

/// Chat statistics
#[derive(Debug, Clone, Default)]
pub struct ChatStats {
    /// Total messages
    pub total_messages: usize,
    /// User messages
    pub user_messages: usize,
    /// Assistant messages
    pub assistant_messages: usize,
    /// Total tokens (estimated)
    pub total_tokens: usize,
    /// Average tokens per message
    pub avg_tokens: f64,
}

impl ChatStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate stats from conversation
    pub fn from_conversation(conv: &Conversation) -> Self {
        let total_messages = conv.len();
        let user_messages = conv.messages.iter().filter(|m| m.role == MessageRole::User).count();
        let assistant_messages = conv
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .count();
        let total_tokens = conv.token_count();
        let avg_tokens = if total_messages > 0 {
            total_tokens as f64 / total_messages as f64
        } else {
            0.0
        };

        Self {
            total_messages,
            user_messages,
            assistant_messages,
            total_tokens,
            avg_tokens,
        }
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Chat: {} messages ({} user, {} assistant), ~{} tokens",
            self.total_messages, self.user_messages, self.assistant_messages, self.total_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_role_default() {
        assert_eq!(MessageRole::default(), MessageRole::User);
    }

    #[test]
    fn test_message_role_as_str() {
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
        assert_eq!(MessageRole::Tool.as_str(), "tool");
    }

    #[test]
    fn test_message_role_from_str() {
        assert_eq!(MessageRole::parse("human"), Some(MessageRole::User));
        assert_eq!(MessageRole::parse("ai"), Some(MessageRole::Assistant));
        assert_eq!(MessageRole::parse("function"), Some(MessageRole::Tool));
        assert_eq!(MessageRole::parse("unknown"), None);
    }

    #[test]
    fn test_message_role_list_all() {
        assert_eq!(MessageRole::list_all().len(), 4);
    }

    #[test]
    fn test_chat_template_format_default() {
        assert_eq!(ChatTemplateFormat::default(), ChatTemplateFormat::Raw);
    }

    #[test]
    fn test_chat_template_format_as_str() {
        assert_eq!(ChatTemplateFormat::ChatML.as_str(), "chatml");
        assert_eq!(ChatTemplateFormat::Llama.as_str(), "llama");
    }

    #[test]
    fn test_chat_template_format_from_str() {
        assert_eq!(ChatTemplateFormat::parse("llama2"), Some(ChatTemplateFormat::Llama));
        assert_eq!(ChatTemplateFormat::parse("none"), Some(ChatTemplateFormat::Raw));
    }

    #[test]
    fn test_chat_template_format_tokens() {
        assert_eq!(ChatTemplateFormat::Llama.bos_token(), "<s>");
        assert_eq!(ChatTemplateFormat::ChatML.eos_token(), "<|im_end|>");
    }

    #[test]
    fn test_chat_message_new() {
        let msg = ChatMessage::new(MessageRole::User, "Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_chat_message_helpers() {
        let sys = ChatMessage::system("You are helpful");
        assert_eq!(sys.role, MessageRole::System);

        let user = ChatMessage::user("Hi");
        assert_eq!(user.role, MessageRole::User);

        let asst = ChatMessage::assistant("Hello!");
        assert_eq!(asst.role, MessageRole::Assistant);

        let tool = ChatMessage::tool("result", "call_123");
        assert_eq!(tool.role, MessageRole::Tool);
        assert_eq!(tool.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_chat_message_with_name() {
        let msg = ChatMessage::user("Hi").with_name("Alice");
        assert_eq!(msg.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_chat_message_token_count() {
        let msg = ChatMessage::user("Hello world"); // 11 chars
        let tokens = msg.token_count();
        assert!(tokens > 0);
    }

    #[test]
    fn test_chat_message_format_raw() {
        let msg = ChatMessage::user("Hello");
        let formatted = msg.format(ChatTemplateFormat::Raw);
        assert_eq!(formatted, "user: Hello");
    }

    #[test]
    fn test_chat_message_format_chatml() {
        let msg = ChatMessage::user("Hello");
        let formatted = msg.format(ChatTemplateFormat::ChatML);
        assert!(formatted.contains("<|im_start|>user"));
        assert!(formatted.contains("<|im_end|>"));
    }

    #[test]
    fn test_chat_message_format_llama() {
        let sys = ChatMessage::system("Be helpful");
        let formatted = sys.format(ChatTemplateFormat::Llama);
        assert!(formatted.contains("<<SYS>>"));
    }

    #[test]
    fn test_conversation_config_default() {
        let config = ConversationConfig::default();
        assert_eq!(config.max_tokens, 4096);
        assert!(config.keep_system);
    }

    #[test]
    fn test_conversation_config_builder() {
        let config = ConversationConfig::new()
            .max_tokens(2048)
            .template(ChatTemplateFormat::ChatML)
            .keep_system(false)
            .truncation(TruncationStrategy::Summarize);

        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.template, ChatTemplateFormat::ChatML);
        assert!(!config.keep_system);
        assert_eq!(config.truncation, TruncationStrategy::Summarize);
    }

    #[test]
    fn test_truncation_strategy_default() {
        assert_eq!(TruncationStrategy::default(), TruncationStrategy::DropOldest);
    }

    #[test]
    fn test_truncation_strategy_from_str() {
        assert_eq!(TruncationStrategy::parse("oldest"), Some(TruncationStrategy::DropOldest));
        assert_eq!(TruncationStrategy::parse("fail"), Some(TruncationStrategy::Error));
    }

    #[test]
    fn test_conversation_new() {
        let conv = Conversation::new();
        assert!(conv.is_empty());
        assert_eq!(conv.len(), 0);
    }

    #[test]
    fn test_conversation_add_messages() {
        let mut conv = Conversation::new();
        conv.add_system("Be helpful");
        conv.add_user("Hello");
        conv.add_assistant("Hi there!");

        assert_eq!(conv.len(), 3);
        assert!(!conv.is_empty());
    }

    #[test]
    fn test_conversation_token_count() {
        let mut conv = Conversation::new();
        conv.add_user("Hello world");
        let tokens = conv.token_count();
        assert!(tokens > 0);
    }

    #[test]
    fn test_conversation_format() {
        let mut conv = Conversation::new();
        conv.add_user("Hi");
        conv.add_assistant("Hello!");

        let formatted = conv.format();
        assert!(formatted.contains("user: Hi"));
        assert!(formatted.contains("assistant: Hello!"));
    }

    #[test]
    fn test_conversation_clear() {
        let mut conv = Conversation::new();
        conv.add_user("Test");
        conv.clear();
        assert!(conv.is_empty());
    }

    #[test]
    fn test_conversation_last() {
        let mut conv = Conversation::new();
        assert!(conv.last().is_none());

        conv.add_user("First");
        conv.add_assistant("Second");
        assert_eq!(conv.last().unwrap().content, "Second");
    }

    #[test]
    fn test_conversation_system_message() {
        let mut conv = Conversation::new();
        conv.add_user("Hi");
        assert!(conv.system_message().is_none());

        conv.add_system("You are helpful");
        assert!(conv.system_message().is_some());
    }

    #[test]
    fn test_chat_stats_default() {
        let stats = ChatStats::default();
        assert_eq!(stats.total_messages, 0);
    }

    #[test]
    fn test_chat_stats_from_conversation() {
        let mut conv = Conversation::new();
        conv.add_user("Hi");
        conv.add_assistant("Hello");
        conv.add_user("How are you?");

        let stats = ChatStats::from_conversation(&conv);
        assert_eq!(stats.total_messages, 3);
        assert_eq!(stats.user_messages, 2);
        assert_eq!(stats.assistant_messages, 1);
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn test_chat_stats_format() {
        let stats = ChatStats {
            total_messages: 5,
            user_messages: 3,
            assistant_messages: 2,
            total_tokens: 100,
            avg_tokens: 20.0,
        };

        let formatted = stats.format();
        assert!(formatted.contains("5 messages"));
        assert!(formatted.contains("3 user"));
        assert!(formatted.contains("2 assistant"));
    }

    #[test]
    fn test_chat_stats_empty_conversation() {
        let conv = Conversation::new();
        let stats = ChatStats::from_conversation(&conv);
        assert_eq!(stats.avg_tokens, 0.0);
    }

    #[test]
    fn test_message_role_parse_system() {
        assert_eq!(MessageRole::parse("system"), Some(MessageRole::System));
    }

    #[test]
    fn test_message_role_parse_bot() {
        assert_eq!(MessageRole::parse("bot"), Some(MessageRole::Assistant));
    }

    #[test]
    fn test_message_role_parse_tool() {
        assert_eq!(MessageRole::parse("tool"), Some(MessageRole::Tool));
    }

    #[test]
    fn test_message_role_parse_user() {
        assert_eq!(MessageRole::parse("user"), Some(MessageRole::User));
    }

    #[test]
    fn test_chat_template_format_parse_chat_ml() {
        assert_eq!(ChatTemplateFormat::parse("chat_ml"), Some(ChatTemplateFormat::ChatML));
    }

    #[test]
    fn test_chat_template_format_parse_chatml() {
        assert_eq!(ChatTemplateFormat::parse("chatml"), Some(ChatTemplateFormat::ChatML));
    }

    #[test]
    fn test_chat_template_format_parse_llama3() {
        assert_eq!(ChatTemplateFormat::parse("llama3"), Some(ChatTemplateFormat::Llama));
    }

    #[test]
    fn test_chat_template_format_parse_gemma() {
        assert_eq!(ChatTemplateFormat::parse("gemma"), Some(ChatTemplateFormat::Gemma));
    }

    #[test]
    fn test_chat_template_format_parse_mistral() {
        assert_eq!(ChatTemplateFormat::parse("mistral"), Some(ChatTemplateFormat::Mistral));
    }

    #[test]
    fn test_chat_template_format_parse_raw() {
        assert_eq!(ChatTemplateFormat::parse("raw"), Some(ChatTemplateFormat::Raw));
    }

    #[test]
    fn test_chat_template_format_parse_unknown() {
        assert_eq!(ChatTemplateFormat::parse("unknown"), None);
    }

    #[test]
    fn test_chat_template_format_list_all() {
        let all = ChatTemplateFormat::list_all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_chat_template_format_as_str_all() {
        assert_eq!(ChatTemplateFormat::Raw.as_str(), "raw");
        assert_eq!(ChatTemplateFormat::Mistral.as_str(), "mistral");
        assert_eq!(ChatTemplateFormat::Gemma.as_str(), "gemma");
    }

    #[test]
    fn test_chat_template_format_bos_all() {
        assert_eq!(ChatTemplateFormat::Raw.bos_token(), "");
        assert_eq!(ChatTemplateFormat::ChatML.bos_token(), "");
        assert_eq!(ChatTemplateFormat::Mistral.bos_token(), "<s>");
        assert_eq!(ChatTemplateFormat::Gemma.bos_token(), "<bos>");
    }

    #[test]
    fn test_chat_template_format_eos_all() {
        assert_eq!(ChatTemplateFormat::Raw.eos_token(), "");
        assert_eq!(ChatTemplateFormat::Llama.eos_token(), "</s>");
        assert_eq!(ChatTemplateFormat::Mistral.eos_token(), "</s>");
        assert_eq!(ChatTemplateFormat::Gemma.eos_token(), "<eos>");
    }

    #[test]
    fn test_chat_message_format_llama_user() {
        let msg = ChatMessage::user("Hello");
        let formatted = msg.format(ChatTemplateFormat::Llama);
        assert!(formatted.contains("[INST]"));
        assert!(formatted.contains("[/INST]"));
    }

    #[test]
    fn test_chat_message_format_llama_assistant() {
        let msg = ChatMessage::assistant("Hi there");
        let formatted = msg.format(ChatTemplateFormat::Llama);
        assert_eq!(formatted, "Hi there");
    }

    #[test]
    fn test_chat_message_format_llama_tool() {
        let msg = ChatMessage::tool("result", "id1");
        let formatted = msg.format(ChatTemplateFormat::Llama);
        assert!(formatted.contains("[TOOL]"));
        assert!(formatted.contains("[/TOOL]"));
    }

    #[test]
    fn test_chat_message_format_mistral_user() {
        let msg = ChatMessage::user("Hello");
        let formatted = msg.format(ChatTemplateFormat::Mistral);
        assert!(formatted.contains("[INST]"));
    }

    #[test]
    fn test_chat_message_format_mistral_assistant() {
        let msg = ChatMessage::assistant("Response");
        let formatted = msg.format(ChatTemplateFormat::Mistral);
        assert_eq!(formatted, "Response");
    }

    #[test]
    fn test_chat_message_format_mistral_system() {
        let msg = ChatMessage::system("Instructions");
        let formatted = msg.format(ChatTemplateFormat::Mistral);
        assert!(formatted.contains("system:"));
    }

    #[test]
    fn test_chat_message_format_mistral_tool() {
        let msg = ChatMessage::tool("data", "call_1");
        let formatted = msg.format(ChatTemplateFormat::Mistral);
        assert!(formatted.contains("tool:"));
    }

    #[test]
    fn test_chat_message_format_gemma() {
        let msg = ChatMessage::user("Hello");
        let formatted = msg.format(ChatTemplateFormat::Gemma);
        assert!(formatted.contains("<start_of_turn>user"));
        assert!(formatted.contains("<end_of_turn>"));
    }

    #[test]
    fn test_truncation_strategy_as_str_all() {
        assert_eq!(TruncationStrategy::DropOldest.as_str(), "drop_oldest");
        assert_eq!(TruncationStrategy::DropNewest.as_str(), "drop_newest");
        assert_eq!(TruncationStrategy::Summarize.as_str(), "summarize");
        assert_eq!(TruncationStrategy::Error.as_str(), "error");
    }

    #[test]
    fn test_truncation_strategy_parse_summary() {
        assert_eq!(TruncationStrategy::parse("summary"), Some(TruncationStrategy::Summarize));
    }

    #[test]
    fn test_truncation_strategy_parse_newest() {
        assert_eq!(TruncationStrategy::parse("newest"), Some(TruncationStrategy::DropNewest));
    }

    #[test]
    fn test_truncation_strategy_parse_error() {
        assert_eq!(TruncationStrategy::parse("error"), Some(TruncationStrategy::Error));
    }

    #[test]
    fn test_truncation_strategy_parse_unknown() {
        assert_eq!(TruncationStrategy::parse("unknown"), None);
    }

    #[test]
    fn test_truncation_strategy_list_all() {
        let all = TruncationStrategy::list_all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_conversation_with_config() {
        let config = ConversationConfig::new()
            .max_tokens(8192)
            .template(ChatTemplateFormat::Llama);
        let conv = Conversation::with_config(config);
        assert_eq!(conv.config.max_tokens, 8192);
        assert_eq!(conv.config.template, ChatTemplateFormat::Llama);
    }

    #[test]
    fn test_conversation_add_raw_message() {
        let mut conv = Conversation::new();
        let msg = ChatMessage::new(MessageRole::Tool, "result").with_name("function1");
        conv.add(msg);
        assert_eq!(conv.len(), 1);
        assert_eq!(conv.last().unwrap().name, Some("function1".to_string()));
    }

    #[test]
    fn test_conversation_format_with_template() {
        let config = ConversationConfig::new().template(ChatTemplateFormat::ChatML);
        let mut conv = Conversation::with_config(config);
        conv.add_user("Hi");

        let formatted = conv.format();
        assert!(formatted.contains("<|im_start|>"));
    }

    #[test]
    fn test_chat_stats_new() {
        let stats = ChatStats::new();
        assert_eq!(stats.total_messages, 0);
        assert_eq!(stats.user_messages, 0);
        assert_eq!(stats.assistant_messages, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.avg_tokens, 0.0);
    }

    #[test]
    fn test_chat_message_clone() {
        let msg = ChatMessage::user("Hello").with_name("User1");
        let cloned = msg.clone();
        assert_eq!(cloned.content, "Hello");
        assert_eq!(cloned.name, Some("User1".to_string()));
    }

    #[test]
    fn test_conversation_clone() {
        let mut conv = Conversation::new();
        conv.add_user("Test");
        let cloned = conv.clone();
        assert_eq!(cloned.len(), 1);
    }

    #[test]
    fn test_conversation_config_clone() {
        let config = ConversationConfig::new().max_tokens(1000);
        let cloned = config.clone();
        assert_eq!(cloned.max_tokens, 1000);
    }

    #[test]
    fn test_chat_stats_clone() {
        let stats = ChatStats {
            total_messages: 10,
            user_messages: 5,
            assistant_messages: 5,
            total_tokens: 200,
            avg_tokens: 20.0,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_messages, 10);
    }

    #[test]
    fn test_message_role_clone() {
        let role = MessageRole::System;
        let cloned = role.clone();
        assert_eq!(cloned, MessageRole::System);
    }

    #[test]
    fn test_chat_template_format_clone() {
        let fmt = ChatTemplateFormat::Llama;
        let cloned = fmt.clone();
        assert_eq!(cloned, ChatTemplateFormat::Llama);
    }

    #[test]
    fn test_truncation_strategy_clone() {
        let strat = TruncationStrategy::Summarize;
        let cloned = strat.clone();
        assert_eq!(cloned, TruncationStrategy::Summarize);
    }

    #[test]
    fn test_message_role_debug() {
        let role = MessageRole::Assistant;
        let debug = format!("{:?}", role);
        assert!(debug.contains("Assistant"));
    }

    #[test]
    fn test_chat_template_format_debug() {
        let fmt = ChatTemplateFormat::ChatML;
        let debug = format!("{:?}", fmt);
        assert!(debug.contains("ChatML"));
    }

    #[test]
    fn test_truncation_strategy_debug() {
        let strat = TruncationStrategy::Error;
        let debug = format!("{:?}", strat);
        assert!(debug.contains("Error"));
    }

    #[test]
    fn test_chat_message_debug() {
        let msg = ChatMessage::user("Test");
        let debug = format!("{:?}", msg);
        assert!(debug.contains("User"));
        assert!(debug.contains("Test"));
    }

    #[test]
    fn test_conversation_debug() {
        let conv = Conversation::new();
        let debug = format!("{:?}", conv);
        assert!(debug.contains("Conversation"));
    }

    #[test]
    fn test_conversation_config_debug() {
        let config = ConversationConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("max_tokens"));
    }

    #[test]
    fn test_chat_stats_debug() {
        let stats = ChatStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("total_messages"));
    }
}
