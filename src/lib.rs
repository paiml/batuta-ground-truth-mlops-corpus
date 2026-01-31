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
//! ## P0 (Core MLOps)
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
//! ## P1 (Hub, Generation, RAG)
//!
//! | Module | Purpose | Python Equivalent |
//! |--------|---------|-------------------|
//! | [`hub`] | Model registry, versioning | `hf_gtc.hub` |
//! | [`generation`] | Text generation, sampling | `hf_gtc.generation` |
//! | [`rag`] | Chunking, retrieval, reranking | `hf_gtc.rag` |
//!
//! ## P2 (Agents, Safety)
//!
//! | Module | Purpose | Python Equivalent |
//! |--------|---------|-------------------|
//! | [`agents`] | Agent workflows, memory, tools | `hf_gtc.agents` |
//! | [`safety`] | Guardrails, privacy, watermarking | `hf_gtc.safety` |
//!
//! ## P3 (Audio, Multimodal)
//!
//! | Module | Purpose | Python Equivalent |
//! |--------|---------|-------------------|
//! | [`audio`] | Speech recognition, audio features | `hf_gtc.audio` |
//! | [`multimodal`] | Vision, document processing | `hf_gtc.multimodal` |
//!
//! # Quick Start
//!
//! ```rust
//! use batuta_ground_truth_mlops_corpus::preprocessing::{Tokenizer, TokenizerConfig};
//! use batuta_ground_truth_mlops_corpus::models::RandomForestConfig;
//! use batuta_ground_truth_mlops_corpus::hub::{RegistryConfig, ModelStage};
//! use batuta_ground_truth_mlops_corpus::generation::{SamplingConfig, SamplingStrategy};
//! use batuta_ground_truth_mlops_corpus::rag::{ChunkConfig, RagConfig};
//! use batuta_ground_truth_mlops_corpus::agents::{AgentConfig, ToolDefinition};
//! use batuta_ground_truth_mlops_corpus::safety::{ContentFilterConfig, PrivacyConfig};
//! use batuta_ground_truth_mlops_corpus::audio::{AudioConfig, SpeechConfig};
//! use batuta_ground_truth_mlops_corpus::multimodal::{VisionConfig, DocumentConfig};
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
//!
//! // Model registry
//! let registry = RegistryConfig::new("my-model").namespace("org");
//! assert!(registry.validate().is_ok());
//!
//! // Text generation
//! let sampling = SamplingConfig::new()
//!     .strategy(SamplingStrategy::TopP)
//!     .temperature(0.7);
//! assert!(sampling.validate().is_ok());
//!
//! // RAG chunking
//! let chunks = ChunkConfig::new().chunk_size(512).overlap(50);
//! assert!(chunks.validate().is_ok());
//!
//! // Agent configuration
//! let agent = AgentConfig::new("assistant")
//!     .tool(ToolDefinition::new("search", "Search the web"))
//!     .max_iterations(10);
//! assert!(agent.validate().is_ok());
//!
//! // Safety guardrails
//! let filter = ContentFilterConfig::new().threshold(0.9);
//! let privacy = PrivacyConfig::new().threshold(0.8);
//! assert!(filter.validate().is_ok());
//! assert!(privacy.validate().is_ok());
//!
//! // Audio processing (Whisper-style)
//! let audio = AudioConfig::whisper().n_mels(80).sample_rate(16000);
//! let speech = SpeechConfig::new().timestamps(true);
//! assert_eq!(audio.n_mels, 80);
//!
//! // Multimodal (vision + document)
//! let vision = VisionConfig::clip().size(224).normalize(true);
//! let doc = DocumentConfig::new().ocr_enabled(true);
//! assert_eq!(vision.size, 224);
//! assert!(doc.ocr_enabled);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// P0: Core MLOps
pub mod preprocessing;
pub mod models;
pub mod training;
pub mod inference;
pub mod evaluation;
pub mod deployment;

// P1: Hub, Generation, RAG
pub mod hub;
pub mod generation;
pub mod rag;

// P2: Agents, Safety
pub mod agents;
pub mod safety;

// P3: Audio, Multimodal
pub mod audio;
pub mod multimodal;

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
        // Verify P0 modules are accessible
        let _ = preprocessing::TokenizerConfig::default();
        let _ = models::RandomForestConfig::default();
        let _ = training::TrainerConfig::default();
        let _ = inference::PipelineConfig::default();
        let _ = evaluation::ClassificationMetrics::default();
        let _ = deployment::ExportConfig::default();
    }

    #[test]
    fn test_p1_module_exports() {
        // Verify P1 modules are accessible
        let _ = hub::RegistryConfig::default();
        let _ = hub::ModelStage::default();
        let _ = generation::SamplingConfig::default();
        let _ = generation::PromptTemplate::default();
        let _ = rag::ChunkConfig::default();
        let _ = rag::RagConfig::default();
    }

    #[test]
    fn test_p2_module_exports() {
        // Verify P2 modules are accessible
        let _ = agents::MemoryType::default();
        let _ = agents::BufferConfig::default();
        let _ = agents::PlanningStrategy::default();
        let _ = agents::ToolType::default();
        let _ = agents::AgentConfig::default();
        let _ = safety::GuardrailType::default();
        let _ = safety::ContentCategory::default();
        let _ = safety::PiiType::default();
        let _ = safety::WatermarkType::default();
    }

    #[test]
    fn test_p3_module_exports() {
        // Verify P3 modules are accessible
        let _ = audio::FeatureType::default();
        let _ = audio::AudioConfig::default();
        let _ = audio::SpeechModel::default();
        let _ = audio::SpeechConfig::default();
        let _ = multimodal::ImageFormat::default();
        let _ = multimodal::VisionConfig::default();
        let _ = multimodal::DocumentType::default();
        let _ = multimodal::DocumentConfig::default();
    }

    #[test]
    fn test_error_display() {
        let err = CorpusError::InvalidInput("test".to_string());
        assert!(err.to_string().contains("Invalid input"));
    }

    #[test]
    fn test_error_display_model_not_found() {
        let err = CorpusError::ModelNotFound("my-model".to_string());
        let display = err.to_string();
        assert!(display.contains("Model not found"));
        assert!(display.contains("my-model"));
    }

    #[test]
    fn test_error_display_tokenization() {
        let err = CorpusError::TokenizationError("invalid utf8".to_string());
        let display = err.to_string();
        assert!(display.contains("Tokenization failed"));
        assert!(display.contains("invalid utf8"));
    }

    #[test]
    fn test_error_display_training() {
        let err = CorpusError::TrainingError("convergence failure".to_string());
        let display = err.to_string();
        assert!(display.contains("Training failed"));
        assert!(display.contains("convergence failure"));
    }

    #[test]
    fn test_error_display_inference() {
        let err = CorpusError::InferenceError("out of memory".to_string());
        let display = err.to_string();
        assert!(display.contains("Inference failed"));
        assert!(display.contains("out of memory"));
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

    #[test]
    fn test_error_debug() {
        let err = CorpusError::InvalidInput("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidInput"));
    }

    #[test]
    fn test_error_clone() {
        let err = CorpusError::ModelNotFound("model".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_eq() {
        let err1 = CorpusError::TrainingError("a".to_string());
        let err2 = CorpusError::TrainingError("a".to_string());
        let err3 = CorpusError::TrainingError("b".to_string());
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_is_error_trait() {
        use std::error::Error;
        let err: Box<dyn Error> = Box::new(CorpusError::InvalidInput("test".to_string()));
        assert!(err.source().is_none());
    }

    #[test]
    fn test_result_type_alias() {
        fn example_fn() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(example_fn().unwrap(), 42);
    }

    #[test]
    fn test_result_error() {
        fn example_err_fn() -> Result<i32> {
            Err(CorpusError::InvalidInput("bad".to_string()))
        }
        assert!(example_err_fn().is_err());
    }
}
