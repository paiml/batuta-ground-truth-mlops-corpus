//! Transformer model configurations
//!
//! Provides configurations for BERT-style transformer models.

/// Configuration for Transformer models
///
/// # Python Equivalent (HuggingFace)
/// ```python
/// from transformers import BertConfig
/// config = BertConfig(hidden_size=768, num_attention_heads=12)
/// ```
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            vocab_size: 30522,
        }
    }
}

impl TransformerConfig {
    /// Create BERT-base configuration
    pub fn bert_base() -> Self {
        Self::default()
    }

    /// Create BERT-large configuration
    pub fn bert_large() -> Self {
        Self {
            hidden_size: 1024,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            vocab_size: 30522,
        }
    }

    /// Set hidden size
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Set number of attention heads
    pub fn num_attention_heads(mut self, heads: usize) -> Self {
        self.num_attention_heads = heads;
        self
    }

    /// Set number of hidden layers
    pub fn num_hidden_layers(mut self, layers: usize) -> Self {
        self.num_hidden_layers = layers;
        self
    }

    /// Set vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_base() {
        let config = TransformerConfig::bert_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_hidden_layers, 12);
    }

    #[test]
    fn test_bert_large() {
        let config = TransformerConfig::bert_large();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_hidden_layers, 24);
    }

    #[test]
    fn test_default() {
        let config = TransformerConfig::default();
        assert_eq!(config.vocab_size, 30522);
    }

    #[test]
    fn test_builder() {
        let config = TransformerConfig::default()
            .hidden_size(512)
            .num_attention_heads(8)
            .num_hidden_layers(6)
            .vocab_size(50000);

        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_hidden_layers, 6);
        assert_eq!(config.vocab_size, 50000);
    }
}
