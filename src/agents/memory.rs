//! Agent Memory Management
//!
//! Working and long-term memory for conversational agents.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.agents import MemoryType, BufferConfig, create_buffer_config
//! config = create_buffer_config(max_messages=100)
//! ```

use std::collections::HashMap;

/// Memory type for agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryType {
    /// Simple buffer memory (default)
    #[default]
    Buffer,
    /// Sliding window memory
    Window,
    /// Summary-based memory
    Summary,
    /// Entity-based memory
    Entity,
    /// Vector store backed memory
    VectorStore,
}

impl MemoryType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Window => "window",
            Self::Summary => "summary",
            Self::Entity => "entity",
            Self::VectorStore => "vector_store",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "buffer" => Some(Self::Buffer),
            "window" | "sliding" => Some(Self::Window),
            "summary" => Some(Self::Summary),
            "entity" => Some(Self::Entity),
            "vector_store" | "vector" => Some(Self::VectorStore),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![Self::Buffer, Self::Window, Self::Summary, Self::Entity, Self::VectorStore]
    }
}

/// Buffer memory configuration
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Maximum messages to store
    pub max_messages: usize,
    /// Maximum tokens to store
    pub max_tokens: usize,
    /// Return messages in reverse order
    pub return_messages: bool,
    /// Include system messages
    pub include_system: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_tokens: 4096,
            return_messages: true,
            include_system: true,
        }
    }
}

impl BufferConfig {
    /// Create new buffer config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max messages
    pub fn max_messages(mut self, n: usize) -> Self {
        self.max_messages = n;
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Set return messages
    pub fn return_messages(mut self, enabled: bool) -> Self {
        self.return_messages = enabled;
        self
    }

    /// Set include system
    pub fn include_system(mut self, enabled: bool) -> Self {
        self.include_system = enabled;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.max_messages == 0 && self.max_tokens == 0 {
            return Err("Must set max_messages or max_tokens".to_string());
        }
        Ok(())
    }
}

/// Window memory configuration
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Window size (number of messages)
    pub window_size: usize,
    /// Overlap between windows
    pub overlap: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            overlap: 2,
        }
    }
}

impl WindowConfig {
    /// Create new window config
    pub fn new(size: usize) -> Self {
        Self {
            window_size: size,
            ..Default::default()
        }
    }

    /// Set overlap
    pub fn overlap(mut self, n: usize) -> Self {
        self.overlap = n;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("Window size must be > 0".to_string());
        }
        if self.overlap >= self.window_size {
            return Err("Overlap must be less than window size".to_string());
        }
        Ok(())
    }
}

/// Summary memory configuration
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Max tokens for summary
    pub max_summary_tokens: usize,
    /// Summarization threshold (messages before summarizing)
    pub threshold: usize,
    /// Preserve recent messages
    pub preserve_recent: usize,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            max_summary_tokens: 500,
            threshold: 20,
            preserve_recent: 5,
        }
    }
}

impl SummaryConfig {
    /// Create new summary config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max summary tokens
    pub fn max_summary_tokens(mut self, n: usize) -> Self {
        self.max_summary_tokens = n;
        self
    }

    /// Set threshold
    pub fn threshold(mut self, n: usize) -> Self {
        self.threshold = n;
        self
    }

    /// Set preserve recent
    pub fn preserve_recent(mut self, n: usize) -> Self {
        self.preserve_recent = n;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.threshold == 0 {
            return Err("Threshold must be > 0".to_string());
        }
        Ok(())
    }
}

/// Entity memory configuration
#[derive(Debug, Clone)]
pub struct EntityConfig {
    /// Entity types to track
    pub entity_types: Vec<String>,
    /// Max entities per type
    pub max_entities: usize,
    /// Update frequency
    pub update_frequency: usize,
}

impl Default for EntityConfig {
    fn default() -> Self {
        Self {
            entity_types: vec!["person".to_string(), "organization".to_string(), "location".to_string()],
            max_entities: 100,
            update_frequency: 5,
        }
    }
}

impl EntityConfig {
    /// Create new entity config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set entity types
    pub fn entity_types(mut self, types: Vec<String>) -> Self {
        self.entity_types = types;
        self
    }

    /// Set max entities
    pub fn max_entities(mut self, n: usize) -> Self {
        self.max_entities = n;
        self
    }

    /// Set update frequency
    pub fn update_frequency(mut self, n: usize) -> Self {
        self.update_frequency = n;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.entity_types.is_empty() {
            return Err("Must specify at least one entity type".to_string());
        }
        Ok(())
    }
}

/// Conversation message
#[derive(Debug, Clone)]
pub struct ConversationMessage {
    /// Role (user, assistant, system)
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp
    pub timestamp: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl ConversationMessage {
    /// Create new message
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set timestamp
    pub fn timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Estimate token count
    pub fn token_count(&self) -> usize {
        self.content.len() / 4 + 4
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total messages stored
    pub total_messages: usize,
    /// Total tokens used
    pub total_tokens: usize,
    /// Memory type
    pub memory_type: String,
    /// Entities tracked (for entity memory)
    pub entities_tracked: usize,
}

impl MemoryStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total messages
    pub fn messages(mut self, n: usize) -> Self {
        self.total_messages = n;
        self
    }

    /// Set total tokens
    pub fn tokens(mut self, n: usize) -> Self {
        self.total_tokens = n;
        self
    }

    /// Set memory type
    pub fn memory_type(mut self, t: impl Into<String>) -> Self {
        self.memory_type = t.into();
        self
    }

    /// Set entities tracked
    pub fn entities(mut self, n: usize) -> Self {
        self.entities_tracked = n;
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Memory [{}]: {} messages, {} tokens",
            self.memory_type, self.total_messages, self.total_tokens
        )
    }
}

/// Calculate memory size in bytes
pub fn calculate_memory_size_bytes(messages: &[ConversationMessage]) -> usize {
    messages.iter().map(|m| m.content.len() + m.role.len() + 32).sum()
}

/// Estimate tokens for messages
pub fn estimate_memory_tokens(messages: &[ConversationMessage]) -> usize {
    messages.iter().map(|m| m.token_count()).sum()
}

/// Calculate window messages to keep
pub fn calculate_window_messages(total: usize, window_size: usize, _overlap: usize) -> usize {
    if total <= window_size {
        total
    } else {
        window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_default() {
        assert_eq!(MemoryType::default(), MemoryType::Buffer);
    }

    #[test]
    fn test_memory_type_as_str() {
        assert_eq!(MemoryType::Buffer.as_str(), "buffer");
        assert_eq!(MemoryType::Window.as_str(), "window");
        assert_eq!(MemoryType::Summary.as_str(), "summary");
        assert_eq!(MemoryType::Entity.as_str(), "entity");
        assert_eq!(MemoryType::VectorStore.as_str(), "vector_store");
    }

    #[test]
    fn test_memory_type_parse() {
        assert_eq!(MemoryType::parse("buffer"), Some(MemoryType::Buffer));
        assert_eq!(MemoryType::parse("sliding"), Some(MemoryType::Window));
        assert_eq!(MemoryType::parse("vector"), Some(MemoryType::VectorStore));
        assert_eq!(MemoryType::parse("unknown"), None);
    }

    #[test]
    fn test_memory_type_list_all() {
        assert_eq!(MemoryType::list_all().len(), 5);
    }

    #[test]
    fn test_buffer_config_default() {
        let config = BufferConfig::default();
        assert_eq!(config.max_messages, 100);
        assert_eq!(config.max_tokens, 4096);
        assert!(config.return_messages);
    }

    #[test]
    fn test_buffer_config_builder() {
        let config = BufferConfig::new()
            .max_messages(50)
            .max_tokens(2048)
            .return_messages(false)
            .include_system(false);

        assert_eq!(config.max_messages, 50);
        assert_eq!(config.max_tokens, 2048);
        assert!(!config.return_messages);
        assert!(!config.include_system);
    }

    #[test]
    fn test_buffer_config_validate() {
        let valid = BufferConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = BufferConfig::new().max_messages(0).max_tokens(0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_window_config_default() {
        let config = WindowConfig::default();
        assert_eq!(config.window_size, 10);
        assert_eq!(config.overlap, 2);
    }

    #[test]
    fn test_window_config_builder() {
        let config = WindowConfig::new(20).overlap(5);
        assert_eq!(config.window_size, 20);
        assert_eq!(config.overlap, 5);
    }

    #[test]
    fn test_window_config_validate() {
        let valid = WindowConfig::default();
        assert!(valid.validate().is_ok());

        let zero_window = WindowConfig::new(0);
        assert!(zero_window.validate().is_err());

        let big_overlap = WindowConfig::new(5).overlap(10);
        assert!(big_overlap.validate().is_err());
    }

    #[test]
    fn test_summary_config_default() {
        let config = SummaryConfig::default();
        assert_eq!(config.max_summary_tokens, 500);
        assert_eq!(config.threshold, 20);
    }

    #[test]
    fn test_summary_config_builder() {
        let config = SummaryConfig::new()
            .max_summary_tokens(1000)
            .threshold(30)
            .preserve_recent(10);

        assert_eq!(config.max_summary_tokens, 1000);
        assert_eq!(config.threshold, 30);
        assert_eq!(config.preserve_recent, 10);
    }

    #[test]
    fn test_summary_config_validate() {
        let valid = SummaryConfig::default();
        assert!(valid.validate().is_ok());

        let zero_threshold = SummaryConfig::new().threshold(0);
        assert!(zero_threshold.validate().is_err());
    }

    #[test]
    fn test_entity_config_default() {
        let config = EntityConfig::default();
        assert_eq!(config.entity_types.len(), 3);
        assert_eq!(config.max_entities, 100);
    }

    #[test]
    fn test_entity_config_builder() {
        let config = EntityConfig::new()
            .entity_types(vec!["custom".to_string()])
            .max_entities(50)
            .update_frequency(10);

        assert_eq!(config.entity_types, vec!["custom"]);
        assert_eq!(config.max_entities, 50);
        assert_eq!(config.update_frequency, 10);
    }

    #[test]
    fn test_entity_config_validate() {
        let valid = EntityConfig::default();
        assert!(valid.validate().is_ok());

        let no_types = EntityConfig::new().entity_types(vec![]);
        assert!(no_types.validate().is_err());
    }

    #[test]
    fn test_conversation_message_new() {
        let msg = ConversationMessage::new("user", "Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_conversation_message_builder() {
        let msg = ConversationMessage::new("assistant", "Hi")
            .timestamp(123456)
            .metadata("source", "chat");

        assert_eq!(msg.timestamp, 123456);
        assert_eq!(msg.metadata.get("source"), Some(&"chat".to_string()));
    }

    #[test]
    fn test_conversation_message_token_count() {
        let msg = ConversationMessage::new("user", "Hello world"); // 11 chars
        let tokens = msg.token_count();
        assert!(tokens > 0);
    }

    #[test]
    fn test_memory_stats_default() {
        let stats = MemoryStats::default();
        assert_eq!(stats.total_messages, 0);
    }

    #[test]
    fn test_memory_stats_builder() {
        let stats = MemoryStats::new()
            .messages(10)
            .tokens(500)
            .memory_type("buffer")
            .entities(5);

        assert_eq!(stats.total_messages, 10);
        assert_eq!(stats.total_tokens, 500);
        assert_eq!(stats.memory_type, "buffer");
        assert_eq!(stats.entities_tracked, 5);
    }

    #[test]
    fn test_memory_stats_format() {
        let stats = MemoryStats::new()
            .messages(5)
            .tokens(100)
            .memory_type("buffer");
        let formatted = stats.format();
        assert!(formatted.contains("buffer"));
        assert!(formatted.contains("5 messages"));
    }

    #[test]
    fn test_calculate_memory_size_bytes() {
        let messages = vec![
            ConversationMessage::new("user", "Hello"),
            ConversationMessage::new("assistant", "Hi there"),
        ];
        let size = calculate_memory_size_bytes(&messages);
        assert!(size > 0);
    }

    #[test]
    fn test_estimate_memory_tokens() {
        let messages = vec![
            ConversationMessage::new("user", "Hello world"),
            ConversationMessage::new("assistant", "Hi"),
        ];
        let tokens = estimate_memory_tokens(&messages);
        assert!(tokens > 0);
    }

    #[test]
    fn test_calculate_window_messages() {
        assert_eq!(calculate_window_messages(5, 10, 2), 5);
        assert_eq!(calculate_window_messages(20, 10, 2), 10);
        assert_eq!(calculate_window_messages(0, 10, 2), 0);
    }
}
