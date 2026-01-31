//! RAG (Retrieval-Augmented Generation) Module
//!
//! Document chunking, retrieval, and reranking for RAG systems.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.rag import (
//!     ChunkingStrategy, create_chunking_config,
//!     RetrievalMethod, RagConfig,
//!     RerankerType, FusionMethod
//! )
//! ```
//!
//! # Example
//!
//! ```rust
//! use batuta_ground_truth_mlops_corpus::rag::{
//!     ChunkConfig, ChunkingStrategy, chunk_document,
//!     RagConfig, RetrievalMethod, DistanceMetric,
//!     FusionMethod, calculate_rrf_score
//! };
//!
//! // Document chunking
//! let config = ChunkConfig::new()
//!     .strategy(ChunkingStrategy::FixedSize)
//!     .chunk_size(512)
//!     .overlap(50);
//! assert!(config.validate().is_ok());
//!
//! let text = "This is a test document. ".repeat(100);
//! let result = chunk_document(&text, &config);
//! assert!(result.total_chunks > 0);
//!
//! // RAG configuration
//! let rag = RagConfig::new()
//!     .method(RetrievalMethod::Hybrid)
//!     .distance_metric(DistanceMetric::Cosine)
//!     .top_k(5);
//! assert!(rag.validate().is_ok());
//!
//! // RRF score calculation
//! let score = calculate_rrf_score(1, 60);
//! assert!(score > 0.0);
//! ```

pub mod chunking;
pub mod retrieval;
pub mod reranking;

pub use chunking::{
    BoundaryDetection,
    Chunk,
    ChunkConfig,
    ChunkResult,
    ChunkingStrategy,
    OverlapType,
    calculate_chunk_count,
    calculate_overlap_ratio,
    chunk_document,
    get_recommended_chunk_size,
};

pub use retrieval::{
    DistanceMetric,
    RagConfig,
    RetrievalMethod,
    RetrievalResult,
    RetrievalStats,
    cosine_similarity,
    dot_product,
    estimate_retrieval_latency,
    euclidean_distance,
};

pub use reranking::{
    Bm25Config,
    FusionConfig,
    FusionMethod,
    RerankerConfig,
    RerankerType,
    RerankResult,
    calculate_bm25_score,
    calculate_linear_fusion,
    calculate_rrf_score,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_module_exports() {
        // Verify all types are accessible
        let _ = ChunkingStrategy::default();
        let _ = RetrievalMethod::default();
        let _ = FusionMethod::default();
    }

    #[test]
    fn test_chunking_integration() {
        let config = ChunkConfig::new()
            .chunk_size(100)
            .overlap(20)
            .min_chunk_size(10);
        
        let text = "Hello world! ".repeat(50);
        let result = chunk_document(&text, &config);
        
        assert!(result.total_chunks > 0);
        assert!(result.avg_chunk_size > 0.0);
    }

    #[test]
    fn test_retrieval_integration() {
        let config = RagConfig::new()
            .method(RetrievalMethod::Hybrid)
            .top_k(10)
            .alpha(0.7);
        
        assert!(config.validate().is_ok());
        assert_eq!(config.alpha, 0.7);
    }

    #[test]
    fn test_reranking_integration() {
        // RRF scores for positions 1-5
        let scores: Vec<f64> = (1..=5).map(|r| calculate_rrf_score(r, 60)).collect();
        
        // Verify monotonically decreasing
        for i in 0..scores.len() - 1 {
            assert!(scores[i] > scores[i + 1]);
        }
    }

    #[test]
    fn test_similarity_functions() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        // Orthogonal vectors have 0 cosine similarity
        assert!((cosine_similarity(&a, &b)).abs() < 0.001);
        
        // Same vectors have 1.0 cosine similarity
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.001);
        
        // Euclidean distance between orthogonal unit vectors
        let dist = euclidean_distance(&a, &b);
        assert!((dist - std::f64::consts::SQRT_2).abs() < 0.001);
    }

    #[test]
    fn test_bm25_scoring() {
        let config = Bm25Config::default();
        
        // Calculate BM25 for a term
        let score = calculate_bm25_score(3.0, 100.0, 10000.0, 150.0, &config);
        assert!(score > 0.0);
    }

    #[test]
    fn test_hybrid_fusion() {
        let dense_score = 0.9;
        let sparse_score = 0.7;
        
        // Equal weights
        let balanced = calculate_linear_fusion(dense_score, sparse_score, 0.5);
        assert!((balanced - 0.8).abs() < 0.001);
        
        // Favor dense
        let dense_heavy = calculate_linear_fusion(dense_score, sparse_score, 0.8);
        assert!(dense_heavy > balanced);
    }
}
