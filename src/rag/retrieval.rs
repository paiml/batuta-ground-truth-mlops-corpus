//! Retrieval Methods
//!
//! Vector and hybrid retrieval for RAG systems.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.rag import RetrievalMethod, DistanceMetric, create_rag_config
//! config = create_rag_config(method=RetrievalMethod.HYBRID, top_k=10)
//! ```

/// Retrieval method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetrievalMethod {
    /// Dense vector retrieval (default)
    #[default]
    Dense,
    /// Sparse (BM25) retrieval
    Sparse,
    /// Hybrid (dense + sparse)
    Hybrid,
    /// Multi-query retrieval
    MultiQuery,
}

impl RetrievalMethod {
    /// Get method name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Hybrid => "hybrid",
            Self::MultiQuery => "multi_query",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "dense" | "vector" | "embedding" => Some(Self::Dense),
            "sparse" | "bm25" | "keyword" => Some(Self::Sparse),
            "hybrid" | "combined" => Some(Self::Hybrid),
            "multi_query" | "multiquery" => Some(Self::MultiQuery),
            _ => None,
        }
    }

    /// List all methods
    pub fn list_all() -> Vec<Self> {
        vec![Self::Dense, Self::Sparse, Self::Hybrid, Self::MultiQuery]
    }
}

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// Cosine similarity (default)
    #[default]
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

impl DistanceMetric {
    /// Get metric name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
            Self::DotProduct => "dot_product",
            Self::Manhattan => "manhattan",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cosine" | "cos" => Some(Self::Cosine),
            "euclidean" | "l2" => Some(Self::Euclidean),
            "dot_product" | "dot" | "ip" => Some(Self::DotProduct),
            "manhattan" | "l1" => Some(Self::Manhattan),
            _ => None,
        }
    }

    /// List all metrics
    pub fn list_all() -> Vec<Self> {
        vec![Self::Cosine, Self::Euclidean, Self::DotProduct, Self::Manhattan]
    }

    /// Higher is better for this metric
    pub fn higher_is_better(&self) -> bool {
        matches!(self, Self::Cosine | Self::DotProduct)
    }
}

/// RAG configuration
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Retrieval method
    pub method: RetrievalMethod,
    /// Distance metric for dense retrieval
    pub distance_metric: DistanceMetric,
    /// Number of documents to retrieve
    pub top_k: usize,
    /// Minimum similarity threshold
    pub min_score: Option<f64>,
    /// Alpha for hybrid search (0 = sparse, 1 = dense)
    pub alpha: f64,
    /// Enable reranking
    pub rerank: bool,
    /// Number of docs to rerank
    pub rerank_top_n: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            method: RetrievalMethod::Dense,
            distance_metric: DistanceMetric::Cosine,
            top_k: 5,
            min_score: None,
            alpha: 0.5,
            rerank: false,
            rerank_top_n: 10,
        }
    }
}

impl RagConfig {
    /// Create new RAG config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set retrieval method
    pub fn method(mut self, method: RetrievalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set top k
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set minimum score threshold
    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Set alpha for hybrid search
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable reranking
    pub fn rerank(mut self, enabled: bool) -> Self {
        self.rerank = enabled;
        self
    }

    /// Set rerank top n
    pub fn rerank_top_n(mut self, n: usize) -> Self {
        self.rerank_top_n = n;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.top_k == 0 {
            return Err("Top-k must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err("Alpha must be in [0, 1]".to_string());
        }
        if self.rerank && self.rerank_top_n < self.top_k {
            return Err("Rerank top-n should be >= top-k".to_string());
        }
        Ok(())
    }
}

/// A retrieval result
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Document/chunk ID
    pub id: String,
    /// Content
    pub content: String,
    /// Similarity/relevance score
    pub score: f64,
    /// Rank (1-indexed)
    pub rank: usize,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl RetrievalResult {
    /// Create new result
    pub fn new(id: impl Into<String>, content: impl Into<String>, score: f64, rank: usize) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            score,
            rank,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Retrieval statistics
#[derive(Debug, Clone, Default)]
pub struct RetrievalStats {
    /// Number of results returned
    pub num_results: usize,
    /// Query time in milliseconds
    pub query_time_ms: f64,
    /// Average score
    pub avg_score: f64,
    /// Max score
    pub max_score: f64,
    /// Min score
    pub min_score: f64,
}

impl RetrievalStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate stats from results
    pub fn from_results(results: &[RetrievalResult], query_time_ms: f64) -> Self {
        if results.is_empty() {
            return Self {
                query_time_ms,
                ..Default::default()
            };
        }

        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);

        Self {
            num_results: results.len(),
            query_time_ms,
            avg_score,
            max_score,
            min_score,
        }
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Retrieved {} results in {:.1}ms (scores: {:.3}-{:.3}, avg: {:.3})",
            self.num_results, self.query_time_ms, self.min_score, self.max_score, self.avg_score
        )
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate euclidean distance between two vectors
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate dot product of two vectors
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Estimate retrieval latency based on index size
pub fn estimate_retrieval_latency(index_size: usize, top_k: usize, use_ann: bool) -> f64 {
    if use_ann {
        // Approximate nearest neighbor: O(log n)
        (index_size as f64).log2() * 0.1 + top_k as f64 * 0.01
    } else {
        // Exact search: O(n)
        index_size as f64 * 0.001 + top_k as f64 * 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_method_default() {
        assert_eq!(RetrievalMethod::default(), RetrievalMethod::Dense);
    }

    #[test]
    fn test_retrieval_method_as_str() {
        assert_eq!(RetrievalMethod::Dense.as_str(), "dense");
        assert_eq!(RetrievalMethod::Sparse.as_str(), "sparse");
        assert_eq!(RetrievalMethod::Hybrid.as_str(), "hybrid");
    }

    #[test]
    fn test_retrieval_method_from_str() {
        assert_eq!(RetrievalMethod::parse("vector"), Some(RetrievalMethod::Dense));
        assert_eq!(RetrievalMethod::parse("bm25"), Some(RetrievalMethod::Sparse));
        assert_eq!(RetrievalMethod::parse("combined"), Some(RetrievalMethod::Hybrid));
        assert_eq!(RetrievalMethod::parse("unknown"), None);
    }

    #[test]
    fn test_retrieval_method_list_all() {
        assert_eq!(RetrievalMethod::list_all().len(), 4);
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_distance_metric_as_str() {
        assert_eq!(DistanceMetric::Cosine.as_str(), "cosine");
        assert_eq!(DistanceMetric::Euclidean.as_str(), "euclidean");
    }

    #[test]
    fn test_distance_metric_from_str() {
        assert_eq!(DistanceMetric::parse("cos"), Some(DistanceMetric::Cosine));
        assert_eq!(DistanceMetric::parse("l2"), Some(DistanceMetric::Euclidean));
        assert_eq!(DistanceMetric::parse("ip"), Some(DistanceMetric::DotProduct));
    }

    #[test]
    fn test_distance_metric_higher_is_better() {
        assert!(DistanceMetric::Cosine.higher_is_better());
        assert!(DistanceMetric::DotProduct.higher_is_better());
        assert!(!DistanceMetric::Euclidean.higher_is_better());
    }

    #[test]
    fn test_rag_config_default() {
        let config = RagConfig::default();
        assert_eq!(config.method, RetrievalMethod::Dense);
        assert_eq!(config.top_k, 5);
        assert_eq!(config.alpha, 0.5);
    }

    #[test]
    fn test_rag_config_builder() {
        let config = RagConfig::new()
            .method(RetrievalMethod::Hybrid)
            .distance_metric(DistanceMetric::DotProduct)
            .top_k(10)
            .min_score(0.5)
            .alpha(0.7)
            .rerank(true)
            .rerank_top_n(20);

        assert_eq!(config.method, RetrievalMethod::Hybrid);
        assert_eq!(config.distance_metric, DistanceMetric::DotProduct);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.min_score, Some(0.5));
        assert_eq!(config.alpha, 0.7);
        assert!(config.rerank);
        assert_eq!(config.rerank_top_n, 20);
    }

    #[test]
    fn test_rag_config_validate() {
        let valid = RagConfig::default();
        assert!(valid.validate().is_ok());

        let zero_k = RagConfig::new().top_k(0);
        assert!(zero_k.validate().is_err());

        let bad_alpha = RagConfig::new().alpha(1.5);
        assert!(bad_alpha.validate().is_err());

        let bad_rerank = RagConfig::new().top_k(10).rerank(true).rerank_top_n(5);
        assert!(bad_rerank.validate().is_err());
    }

    #[test]
    fn test_retrieval_result_new() {
        let result = RetrievalResult::new("doc1", "Hello world", 0.95, 1);
        assert_eq!(result.id, "doc1");
        assert_eq!(result.content, "Hello world");
        assert_eq!(result.score, 0.95);
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn test_retrieval_result_with_metadata() {
        let result = RetrievalResult::new("doc1", "test", 0.9, 1)
            .with_metadata("source", "file.txt");
        assert_eq!(result.metadata.get("source"), Some(&"file.txt".to_string()));
    }

    #[test]
    fn test_retrieval_stats_default() {
        let stats = RetrievalStats::default();
        assert_eq!(stats.num_results, 0);
    }

    #[test]
    fn test_retrieval_stats_from_results() {
        let results = vec![
            RetrievalResult::new("1", "a", 0.9, 1),
            RetrievalResult::new("2", "b", 0.8, 2),
            RetrievalResult::new("3", "c", 0.7, 3),
        ];
        
        let stats = RetrievalStats::from_results(&results, 10.0);
        assert_eq!(stats.num_results, 3);
        assert_eq!(stats.query_time_ms, 10.0);
        assert_eq!(stats.max_score, 0.9);
        assert_eq!(stats.min_score, 0.7);
        assert!((stats.avg_score - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_retrieval_stats_from_empty() {
        let stats = RetrievalStats::from_results(&[], 5.0);
        assert_eq!(stats.num_results, 0);
        assert_eq!(stats.query_time_ms, 5.0);
    }

    #[test]
    fn test_retrieval_stats_format() {
        let stats = RetrievalStats {
            num_results: 5,
            query_time_ms: 12.5,
            avg_score: 0.85,
            max_score: 0.95,
            min_score: 0.75,
        };
        let formatted = stats.format();
        assert!(formatted.contains("5 results"));
        assert!(formatted.contains("12.5ms"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance_edge_cases() {
        assert_eq!(euclidean_distance(&[1.0], &[1.0, 2.0]), f64::INFINITY);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_dot_product_edge_cases() {
        assert_eq!(dot_product(&[], &[]), 0.0);
        assert_eq!(dot_product(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_estimate_retrieval_latency() {
        let ann_latency = estimate_retrieval_latency(1000000, 10, true);
        let exact_latency = estimate_retrieval_latency(1000000, 10, false);
        
        assert!(ann_latency < exact_latency);
        assert!(ann_latency > 0.0);
    }
}
