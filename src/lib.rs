//! Batuta Ground Truth MLOps Corpus
//!
//! Production-ready Rust MLOps patterns for the Sovereign AI Stack.
//! This crate provides reference implementations that mirror
//! [hf-ground-truth-corpus](https://github.com/paiml/hugging-face-ground-truth-corpus)
//! but use pure Rust with trueno and aprender.
//!
//! # Zero External Dependencies
//!
//! This corpus uses ONLY the Sovereign AI Stack:
//! - `trueno` - SIMD tensor operations
//! - `aprender` - ML algorithms
//!
//! # Module Overview
//!
//! | Module | Purpose | Python Equivalent |
//! |--------|---------|-------------------|
//! | [`preprocessing`] | Text/data preprocessing | `hf_gtc.preprocessing` |
//! | [`models`] | Model configurations | `sklearn`, `hf_gtc.models` |
//! | [`training`] | Training utilities | `hf_gtc.training` |
//! | [`inference`] | Inference pipelines | `hf_gtc.inference` |
//! | [`evaluation`] | Metrics and evaluation | `hf_gtc.evaluation` |
//! | [`deployment`] | Export and quantization | `hf_gtc.deployment` |
//!
//! # Quick Start
//!
//! ```rust
//! use batuta_ground_truth_mlops_corpus::preprocessing::{Tokenizer, TokenizerConfig};
//! use batuta_ground_truth_mlops_corpus::models::RandomForestConfig;
//!
//! // Tokenization
//! let tokenizer = Tokenizer::new(TokenizerConfig::default());
//! let tokens = tokenizer.tokenize("Hello, world!");
//! assert!(!tokens.is_empty());
//!
//! // Random Forest config (sklearn-compatible)
//! let config = RandomForestConfig::default()
//!     .n_estimators(100)
//!     .max_depth(Some(10));
//! assert_eq!(config.n_estimators, 100);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod preprocessing;
pub mod models;
pub mod training;
pub mod inference;
pub mod evaluation;
pub mod deployment;

/// Common error types for the corpus
pub mod error {
    /// Errors that can occur in corpus operations
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CorpusError {
        /// Invalid input data
        InvalidInput(String),
        /// Model not found
        ModelNotFound(String),
        /// Tokenization error
        TokenizationError(String),
        /// Training error
        TrainingError(String),
        /// Inference error
        InferenceError(String),
    }

    impl std::fmt::Display for CorpusError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
                Self::ModelNotFound(msg) => write!(f, "Model not found: {}", msg),
                Self::TokenizationError(msg) => write!(f, "Tokenization failed: {}", msg),
                Self::TrainingError(msg) => write!(f, "Training failed: {}", msg),
                Self::InferenceError(msg) => write!(f, "Inference failed: {}", msg),
            }
        }
    }

    impl std::error::Error for CorpusError {}

    /// Result type alias for corpus operations
    pub type Result<T> = std::result::Result<T, CorpusError>;
}

pub use error::{CorpusError, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all modules are accessible
        let _ = preprocessing::TokenizerConfig::default();
        let _ = models::RandomForestConfig::default();
    }

    #[test]
    fn test_error_display() {
        let err = CorpusError::InvalidInput("test".to_string());
        assert!(err.to_string().contains("Invalid input"));
    }

    #[test]
    fn test_error_variants() {
        let errors = vec![
            CorpusError::InvalidInput("a".into()),
            CorpusError::ModelNotFound("b".into()),
            CorpusError::TokenizationError("c".into()),
            CorpusError::TrainingError("d".into()),
            CorpusError::InferenceError("e".into()),
        ];
        assert_eq!(errors.len(), 5);
    }
}
