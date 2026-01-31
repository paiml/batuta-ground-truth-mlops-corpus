//! Text tokenization utilities
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.preprocessing.tokenization import preprocess_text, Tokenizer
//! tokens = preprocess_text("Hello, world!")
//! ```
//!
//! # Rust Equivalent
//! ```rust
//! use batuta_ground_truth_mlops_corpus::preprocessing::{Tokenizer, TokenizerConfig};
//!
//! let tokenizer = Tokenizer::new(TokenizerConfig::default());
//! let tokens = tokenizer.tokenize("Hello, world!");
//! ```

/// A single token with metadata
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Token text
    pub text: String,
    /// Token ID (if using vocabulary)
    pub id: Option<u32>,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
}

impl Token {
    /// Create a new token
    pub fn new(text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            id: None,
            start,
            end,
        }
    }

    /// Create a token with an ID
    pub fn with_id(mut self, id: u32) -> Self {
        self.id = Some(id);
        self
    }
}

/// Tokenizer type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenizerType {
    /// Whitespace tokenization
    #[default]
    Whitespace,
    /// WordPiece tokenization (BERT-style)
    WordPiece,
    /// Byte-Pair Encoding (GPT-style)
    Bpe,
}

/// Configuration for tokenizer
///
/// # Python Equivalent
/// ```python
/// from transformers import AutoTokenizer
/// tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
/// ```
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
    /// Whether to lowercase text
    pub lowercase: bool,
    /// Maximum sequence length
    pub max_length: usize,
    /// Truncation strategy
    pub truncation: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_type: TokenizerType::Whitespace,
            lowercase: true,
            max_length: 512,
            truncation: true,
        }
    }
}

impl TokenizerConfig {
    /// Create a BERT-style tokenizer config
    pub fn bert() -> Self {
        Self {
            tokenizer_type: TokenizerType::WordPiece,
            lowercase: true,
            max_length: 512,
            truncation: true,
        }
    }

    /// Create a GPT-style tokenizer config
    pub fn gpt() -> Self {
        Self {
            tokenizer_type: TokenizerType::Bpe,
            lowercase: false,
            max_length: 1024,
            truncation: true,
        }
    }

    /// Set maximum sequence length
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = len;
        self
    }

    /// Enable/disable lowercase
    pub fn lowercase(mut self, enable: bool) -> Self {
        self.lowercase = enable;
        self
    }

    /// Set tokenizer type
    pub fn tokenizer_type(mut self, t: TokenizerType) -> Self {
        self.tokenizer_type = t;
        self
    }

    /// Enable/disable truncation
    pub fn truncation(mut self, enable: bool) -> Self {
        self.truncation = enable;
        self
    }
}

/// Text tokenizer
///
/// # Python Equivalent
/// ```python
/// from hf_gtc.preprocessing.tokenization import Tokenizer
/// tokenizer = Tokenizer(vocab_size=30000)
/// tokens = tokenizer.encode("Hello, world!")
/// ```
#[derive(Debug, Clone)]
pub struct Tokenizer {
    config: TokenizerConfig,
}

impl Tokenizer {
    /// Create a new tokenizer with the given configuration
    pub fn new(config: TokenizerConfig) -> Self {
        Self { config }
    }

    /// Tokenize text into tokens
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let processed = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let tokens = match self.config.tokenizer_type {
            TokenizerType::Whitespace => self.whitespace_tokenize(&processed),
            TokenizerType::WordPiece => self.wordpiece_tokenize(&processed),
            TokenizerType::Bpe => self.whitespace_tokenize(&processed), // Simplified
        };

        if self.config.truncation && tokens.len() > self.config.max_length {
            tokens.into_iter().take(self.config.max_length).collect()
        } else {
            tokens
        }
    }

    /// Whitespace tokenization
    fn whitespace_tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut start = 0;

        for word in text.split_whitespace() {
            if let Some(pos) = text[start..].find(word) {
                let word_start = start + pos;
                let word_end = word_start + word.len();
                tokens.push(Token::new(word, word_start, word_end));
                start = word_end;
            }
        }

        tokens
    }

    /// WordPiece tokenization (BERT-style)
    fn wordpiece_tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current_pos = 0;

        for word in text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation()) {
            if word.is_empty() {
                continue;
            }

            if let Some(pos) = text[current_pos..].find(word) {
                let start = current_pos + pos;
                let end = start + word.len();

                // Split long words into subwords
                if word.len() > 4 {
                    let mid = word.len() / 2;
                    tokens.push(Token::new(&word[..mid], start, start + mid));
                    tokens.push(Token::new(format!("##{}", &word[mid..]), start + mid, end));
                } else {
                    tokens.push(Token::new(word, start, end));
                }
                current_pos = end;
            }
        }

        tokens
    }

    /// Get the configuration
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(TokenizerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token::new("hello", 0, 5);
        assert_eq!(token.text, "hello");
        assert_eq!(token.start, 0);
        assert_eq!(token.end, 5);
        assert!(token.id.is_none());
    }

    #[test]
    fn test_token_with_id() {
        let token = Token::new("hello", 0, 5).with_id(42);
        assert_eq!(token.id, Some(42));
    }

    #[test]
    fn test_whitespace_tokenize() {
        let tokenizer = Tokenizer::new(TokenizerConfig::default());
        let tokens = tokenizer.tokenize("Hello world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello"); // lowercase
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_bert_config() {
        let config = TokenizerConfig::bert();
        assert_eq!(config.tokenizer_type, TokenizerType::WordPiece);
        assert!(config.lowercase);
    }

    #[test]
    fn test_gpt_config() {
        let config = TokenizerConfig::gpt();
        assert_eq!(config.tokenizer_type, TokenizerType::Bpe);
        assert!(!config.lowercase);
    }

    #[test]
    fn test_truncation() {
        let config = TokenizerConfig::default().max_length(2);
        let tokenizer = Tokenizer::new(config);
        let tokens = tokenizer.tokenize("one two three four");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_empty_input() {
        let tokenizer = Tokenizer::new(TokenizerConfig::default());
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_wordpiece_subwords() {
        let tokenizer = Tokenizer::new(TokenizerConfig::bert());
        let tokens = tokenizer.tokenize("understanding");
        assert!(tokens.len() >= 1);
    }

    #[test]
    fn test_no_lowercase() {
        let config = TokenizerConfig::default().lowercase(false);
        let tokenizer = Tokenizer::new(config);
        let tokens = tokenizer.tokenize("Hello World");
        assert_eq!(tokens[0].text, "Hello");
    }

    #[test]
    fn test_default_tokenizer() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.tokenize("test");
        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_config_builder() {
        let config = TokenizerConfig::default()
            .max_length(100)
            .lowercase(false)
            .truncation(false)
            .tokenizer_type(TokenizerType::WordPiece);

        assert_eq!(config.max_length, 100);
        assert!(!config.lowercase);
        assert!(!config.truncation);
        assert_eq!(config.tokenizer_type, TokenizerType::WordPiece);
    }

    #[test]
    fn test_token_positions_valid() {
        let tokenizer = Tokenizer::default();
        let text = "hello world test";
        let tokens = tokenizer.tokenize(text);

        for token in &tokens {
            assert!(token.start <= token.end);
            assert!(token.end <= text.len());
        }
    }

    #[test]
    fn test_multiple_spaces() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.tokenize("hello    world");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_punctuation_handling() {
        let tokenizer = Tokenizer::new(TokenizerConfig::bert());
        let tokens = tokenizer.tokenize("hello, world!");
        assert!(!tokens.is_empty());
    }
}
