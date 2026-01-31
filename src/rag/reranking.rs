//! Reranking Methods
//!
//! Rerank retrieval results for improved relevance.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.rag import RerankerType, FusionMethod, calculate_rrf_score
//! score = calculate_rrf_score(rank=1, k=60)
//! ```

/// Reranker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankerType {
    /// No reranking (default)
    #[default]
    None,
    /// Cross-encoder reranking
    CrossEncoder,
    /// ColBERT late interaction
    ColBERT,
    /// LLM-based reranking
    LLM,
    /// Cohere reranker
    Cohere,
}

impl RerankerType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::CrossEncoder => "cross_encoder",
            Self::ColBERT => "colbert",
            Self::LLM => "llm",
            Self::Cohere => "cohere",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "cross_encoder" | "crossencoder" | "ce" => Some(Self::CrossEncoder),
            "colbert" => Some(Self::ColBERT),
            "llm" | "gpt" => Some(Self::LLM),
            "cohere" => Some(Self::Cohere),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![Self::None, Self::CrossEncoder, Self::ColBERT, Self::LLM, Self::Cohere]
    }
}

/// Fusion method for combining rankings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion (default)
    #[default]
    RRF,
    /// Linear combination
    Linear,
    /// CombSUM
    CombSum,
    /// CombMNZ
    CombMnz,
}

impl FusionMethod {
    /// Get method name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RRF => "rrf",
            Self::Linear => "linear",
            Self::CombSum => "combsum",
            Self::CombMnz => "combmnz",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rrf" | "reciprocal" => Some(Self::RRF),
            "linear" | "weighted" => Some(Self::Linear),
            "combsum" | "sum" => Some(Self::CombSum),
            "combmnz" | "mnz" => Some(Self::CombMnz),
            _ => None,
        }
    }

    /// List all methods
    pub fn list_all() -> Vec<Self> {
        vec![Self::RRF, Self::Linear, Self::CombSum, Self::CombMnz]
    }
}

/// Reranker configuration
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Reranker type
    pub reranker_type: RerankerType,
    /// Model name/path for reranker
    pub model: String,
    /// Top-n to rerank
    pub top_n: usize,
    /// Batch size for reranking
    pub batch_size: usize,
    /// Normalize scores
    pub normalize_scores: bool,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            reranker_type: RerankerType::None,
            model: String::new(),
            top_n: 10,
            batch_size: 8,
            normalize_scores: true,
        }
    }
}

impl RerankerConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reranker type
    pub fn reranker_type(mut self, rt: RerankerType) -> Self {
        self.reranker_type = rt;
        self
    }

    /// Set model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set top-n
    pub fn top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set normalize scores
    pub fn normalize_scores(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.top_n == 0 {
            return Err("Top-n must be > 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("Batch size must be > 0".to_string());
        }
        if self.reranker_type != RerankerType::None && self.model.is_empty() {
            return Err("Model required for non-None reranker".to_string());
        }
        Ok(())
    }
}

/// Fusion configuration
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fusion method
    pub method: FusionMethod,
    /// Weight for dense scores (hybrid)
    pub dense_weight: f64,
    /// Weight for sparse scores (hybrid)
    pub sparse_weight: f64,
    /// RRF k parameter
    pub rrf_k: usize,
    /// Normalize before fusion
    pub normalize: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::RRF,
            dense_weight: 0.5,
            sparse_weight: 0.5,
            rrf_k: 60,
            normalize: true,
        }
    }
}

impl FusionConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set fusion method
    pub fn method(mut self, method: FusionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set dense weight
    pub fn dense_weight(mut self, weight: f64) -> Self {
        self.dense_weight = weight;
        self
    }

    /// Set sparse weight
    pub fn sparse_weight(mut self, weight: f64) -> Self {
        self.sparse_weight = weight;
        self
    }

    /// Set RRF k
    pub fn rrf_k(mut self, k: usize) -> Self {
        self.rrf_k = k;
        self
    }

    /// Set normalize
    pub fn normalize(mut self, norm: bool) -> Self {
        self.normalize = norm;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.dense_weight < 0.0 || self.sparse_weight < 0.0 {
            return Err("Weights must be non-negative".to_string());
        }
        if self.rrf_k == 0 {
            return Err("RRF k must be > 0".to_string());
        }
        Ok(())
    }
}

/// Calculate Reciprocal Rank Fusion score
pub fn calculate_rrf_score(rank: usize, k: usize) -> f64 {
    if rank == 0 {
        return 0.0;
    }
    1.0 / (k as f64 + rank as f64)
}

/// Calculate linear fusion score
pub fn calculate_linear_fusion(dense_score: f64, sparse_score: f64, alpha: f64) -> f64 {
    alpha * dense_score + (1.0 - alpha) * sparse_score
}

/// Reranked result
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Document ID
    pub id: String,
    /// Original rank
    pub original_rank: usize,
    /// New rank after reranking
    pub new_rank: usize,
    /// Original score
    pub original_score: f64,
    /// New score after reranking
    pub new_score: f64,
    /// Content
    pub content: String,
}

impl RerankResult {
    /// Create new result
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        original_rank: usize,
        new_rank: usize,
        original_score: f64,
        new_score: f64,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            original_rank,
            new_rank,
            original_score,
            new_score,
        }
    }

    /// Calculate rank change
    pub fn rank_change(&self) -> i32 {
        self.original_rank as i32 - self.new_rank as i32
    }
}

/// BM25 configuration
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// k1 parameter (term frequency saturation)
    pub k1: f64,
    /// b parameter (document length normalization)
    pub b: f64,
    /// Average document length
    pub avg_doc_len: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            avg_doc_len: 100.0,
        }
    }
}

impl Bm25Config {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set k1
    pub fn k1(mut self, k1: f64) -> Self {
        self.k1 = k1;
        self
    }

    /// Set b
    pub fn b(mut self, b: f64) -> Self {
        self.b = b;
        self
    }

    /// Set average document length
    pub fn avg_doc_len(mut self, len: f64) -> Self {
        self.avg_doc_len = len;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.k1 < 0.0 {
            return Err("k1 must be non-negative".to_string());
        }
        if !(0.0..=1.0).contains(&self.b) {
            return Err("b must be in [0, 1]".to_string());
        }
        if self.avg_doc_len <= 0.0 {
            return Err("avg_doc_len must be > 0".to_string());
        }
        Ok(())
    }
}

/// Calculate BM25 score for a term
pub fn calculate_bm25_score(
    tf: f64,
    df: f64,
    num_docs: f64,
    doc_len: f64,
    config: &Bm25Config,
) -> f64 {
    let idf = ((num_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
    let tf_component = (tf * (config.k1 + 1.0))
        / (tf + config.k1 * (1.0 - config.b + config.b * doc_len / config.avg_doc_len));
    idf * tf_component
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranker_type_default() {
        assert_eq!(RerankerType::default(), RerankerType::None);
    }

    #[test]
    fn test_reranker_type_as_str() {
        assert_eq!(RerankerType::CrossEncoder.as_str(), "cross_encoder");
        assert_eq!(RerankerType::ColBERT.as_str(), "colbert");
    }

    #[test]
    fn test_reranker_type_from_str() {
        assert_eq!(RerankerType::parse("ce"), Some(RerankerType::CrossEncoder));
        assert_eq!(RerankerType::parse("gpt"), Some(RerankerType::LLM));
        assert_eq!(RerankerType::parse("unknown"), None);
    }

    #[test]
    fn test_reranker_type_list_all() {
        assert_eq!(RerankerType::list_all().len(), 5);
    }

    #[test]
    fn test_fusion_method_default() {
        assert_eq!(FusionMethod::default(), FusionMethod::RRF);
    }

    #[test]
    fn test_fusion_method_as_str() {
        assert_eq!(FusionMethod::RRF.as_str(), "rrf");
        assert_eq!(FusionMethod::Linear.as_str(), "linear");
    }

    #[test]
    fn test_fusion_method_from_str() {
        assert_eq!(FusionMethod::parse("reciprocal"), Some(FusionMethod::RRF));
        assert_eq!(FusionMethod::parse("weighted"), Some(FusionMethod::Linear));
        assert_eq!(FusionMethod::parse("mnz"), Some(FusionMethod::CombMnz));
    }

    #[test]
    fn test_fusion_method_list_all() {
        assert_eq!(FusionMethod::list_all().len(), 4);
    }

    #[test]
    fn test_reranker_config_default() {
        let config = RerankerConfig::default();
        assert_eq!(config.reranker_type, RerankerType::None);
        assert_eq!(config.top_n, 10);
        assert!(config.normalize_scores);
    }

    #[test]
    fn test_reranker_config_builder() {
        let config = RerankerConfig::new()
            .reranker_type(RerankerType::CrossEncoder)
            .model("cross-encoder/ms-marco")
            .top_n(20)
            .batch_size(16)
            .normalize_scores(false);

        assert_eq!(config.reranker_type, RerankerType::CrossEncoder);
        assert_eq!(config.model, "cross-encoder/ms-marco");
        assert_eq!(config.top_n, 20);
        assert_eq!(config.batch_size, 16);
        assert!(!config.normalize_scores);
    }

    #[test]
    fn test_reranker_config_validate() {
        let valid_none = RerankerConfig::default();
        assert!(valid_none.validate().is_ok());

        let valid_with_model = RerankerConfig::new()
            .reranker_type(RerankerType::CrossEncoder)
            .model("model");
        assert!(valid_with_model.validate().is_ok());

        let zero_top_n = RerankerConfig::new().top_n(0);
        assert!(zero_top_n.validate().is_err());

        let no_model = RerankerConfig::new().reranker_type(RerankerType::CrossEncoder);
        assert!(no_model.validate().is_err());
    }

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert_eq!(config.method, FusionMethod::RRF);
        assert_eq!(config.rrf_k, 60);
        assert_eq!(config.dense_weight, 0.5);
    }

    #[test]
    fn test_fusion_config_builder() {
        let config = FusionConfig::new()
            .method(FusionMethod::Linear)
            .dense_weight(0.7)
            .sparse_weight(0.3)
            .rrf_k(40)
            .normalize(false);

        assert_eq!(config.method, FusionMethod::Linear);
        assert_eq!(config.dense_weight, 0.7);
        assert_eq!(config.sparse_weight, 0.3);
        assert_eq!(config.rrf_k, 40);
        assert!(!config.normalize);
    }

    #[test]
    fn test_fusion_config_validate() {
        let valid = FusionConfig::default();
        assert!(valid.validate().is_ok());

        let neg_weight = FusionConfig::new().dense_weight(-0.5);
        assert!(neg_weight.validate().is_err());

        let zero_k = FusionConfig::new().rrf_k(0);
        assert!(zero_k.validate().is_err());
    }

    #[test]
    fn test_calculate_rrf_score() {
        let score1 = calculate_rrf_score(1, 60);
        let score2 = calculate_rrf_score(2, 60);
        let score10 = calculate_rrf_score(10, 60);

        assert!(score1 > score2);
        assert!(score2 > score10);
        assert!((score1 - 1.0 / 61.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_rrf_score_zero_rank() {
        assert_eq!(calculate_rrf_score(0, 60), 0.0);
    }

    #[test]
    fn test_calculate_linear_fusion() {
        let score = calculate_linear_fusion(0.8, 0.6, 0.5);
        assert!((score - 0.7).abs() < 0.001);

        let dense_only = calculate_linear_fusion(0.8, 0.6, 1.0);
        assert!((dense_only - 0.8).abs() < 0.001);

        let sparse_only = calculate_linear_fusion(0.8, 0.6, 0.0);
        assert!((sparse_only - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_rerank_result_new() {
        let result = RerankResult::new("doc1", "content", 5, 2, 0.7, 0.9);
        assert_eq!(result.id, "doc1");
        assert_eq!(result.original_rank, 5);
        assert_eq!(result.new_rank, 2);
        assert_eq!(result.original_score, 0.7);
        assert_eq!(result.new_score, 0.9);
    }

    #[test]
    fn test_rerank_result_rank_change() {
        let improved = RerankResult::new("doc1", "c", 5, 2, 0.5, 0.9);
        assert_eq!(improved.rank_change(), 3); // moved up

        let worsened = RerankResult::new("doc2", "c", 2, 5, 0.9, 0.5);
        assert_eq!(worsened.rank_change(), -3); // moved down

        let unchanged = RerankResult::new("doc3", "c", 3, 3, 0.7, 0.7);
        assert_eq!(unchanged.rank_change(), 0);
    }

    #[test]
    fn test_bm25_config_default() {
        let config = Bm25Config::default();
        assert_eq!(config.k1, 1.5);
        assert_eq!(config.b, 0.75);
        assert_eq!(config.avg_doc_len, 100.0);
    }

    #[test]
    fn test_bm25_config_builder() {
        let config = Bm25Config::new()
            .k1(1.2)
            .b(0.8)
            .avg_doc_len(200.0);

        assert_eq!(config.k1, 1.2);
        assert_eq!(config.b, 0.8);
        assert_eq!(config.avg_doc_len, 200.0);
    }

    #[test]
    fn test_bm25_config_validate() {
        let valid = Bm25Config::default();
        assert!(valid.validate().is_ok());

        let neg_k1 = Bm25Config::new().k1(-0.5);
        assert!(neg_k1.validate().is_err());

        let bad_b = Bm25Config::new().b(1.5);
        assert!(bad_b.validate().is_err());

        let zero_len = Bm25Config::new().avg_doc_len(0.0);
        assert!(zero_len.validate().is_err());
    }

    #[test]
    fn test_calculate_bm25_score() {
        let config = Bm25Config::default();
        let score = calculate_bm25_score(2.0, 10.0, 1000.0, 100.0, &config);
        assert!(score > 0.0);

        // Higher TF should give higher score
        let score_high_tf = calculate_bm25_score(5.0, 10.0, 1000.0, 100.0, &config);
        assert!(score_high_tf > score);

        // Lower DF should give higher score (rarer term)
        let score_low_df = calculate_bm25_score(2.0, 5.0, 1000.0, 100.0, &config);
        assert!(score_low_df > score);
    }
}
