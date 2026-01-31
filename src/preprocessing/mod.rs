//! Preprocessing module - Text and data preprocessing utilities

mod tokenization;
mod normalization;
mod augmentation;

pub use tokenization::{Token, Tokenizer, TokenizerConfig, TokenizerType};
pub use normalization::{Normalizer, NormalizerConfig};
pub use augmentation::{Augmenter, AugmentationConfig, AugmentationType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = Tokenizer::default();
        let tokens = tokenizer.tokenize("Hello, world!");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_normalizer_basic() {
        let normalizer = Normalizer::new(NormalizerConfig {
            lowercase: true,
            collapse_whitespace: true,
        });
        assert_eq!(normalizer.normalize("  HELLO  World  "), "hello world");
    }
}
