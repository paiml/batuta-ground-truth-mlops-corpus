//! Model Watermarking
//!
//! Watermarking techniques for model ownership and tracing.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.safety import WatermarkType, WatermarkConfig, embed_watermark
//! config = WatermarkConfig(type=WatermarkType.SOFT, strength=0.5)
//! ```

/// Watermark type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WatermarkType {
    /// Soft watermark (vocabulary bias, default)
    #[default]
    Soft,
    /// Hard watermark (deterministic)
    Hard,
    /// Semantic watermark
    Semantic,
    /// Statistical watermark
    Statistical,
    /// Neural watermark
    Neural,
    /// Cryptographic watermark
    Cryptographic,
}

impl WatermarkType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Soft => "soft",
            Self::Hard => "hard",
            Self::Semantic => "semantic",
            Self::Statistical => "statistical",
            Self::Neural => "neural",
            Self::Cryptographic => "cryptographic",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "soft" | "bias" => Some(Self::Soft),
            "hard" | "deterministic" => Some(Self::Hard),
            "semantic" | "meaning" => Some(Self::Semantic),
            "statistical" | "stats" => Some(Self::Statistical),
            "neural" | "nn" => Some(Self::Neural),
            "cryptographic" | "crypto" => Some(Self::Cryptographic),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Soft,
            Self::Hard,
            Self::Semantic,
            Self::Statistical,
            Self::Neural,
            Self::Cryptographic,
        ]
    }

    /// Check if watermark is detectable without key
    pub fn requires_key(&self) -> bool {
        matches!(self, Self::Cryptographic | Self::Neural)
    }

    /// Get robustness level (1-10)
    pub fn robustness(&self) -> u8 {
        match self {
            Self::Soft => 4,
            Self::Hard => 6,
            Self::Semantic => 7,
            Self::Statistical => 5,
            Self::Neural => 8,
            Self::Cryptographic => 9,
        }
    }
}

/// Detection method for watermarks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectionMethod {
    /// Statistical analysis (default)
    #[default]
    Statistical,
    /// Key-based verification
    KeyBased,
    /// Pattern matching
    PatternMatch,
    /// Neural network detection
    NeuralNet,
    /// Hybrid approach
    Hybrid,
}

impl DetectionMethod {
    /// Get method name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Statistical => "statistical",
            Self::KeyBased => "key_based",
            Self::PatternMatch => "pattern_match",
            Self::NeuralNet => "neural_net",
            Self::Hybrid => "hybrid",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "statistical" | "stats" => Some(Self::Statistical),
            "key_based" | "key" => Some(Self::KeyBased),
            "pattern_match" | "pattern" => Some(Self::PatternMatch),
            "neural_net" | "neural" | "nn" => Some(Self::NeuralNet),
            "hybrid" | "combined" => Some(Self::Hybrid),
            _ => None,
        }
    }

    /// Check if requires model
    pub fn requires_model(&self) -> bool {
        matches!(self, Self::NeuralNet | Self::Hybrid)
    }
}

/// Watermark embedding configuration
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Watermark type
    pub watermark_type: WatermarkType,
    /// Watermark strength (0.0-1.0)
    pub strength: f32,
    /// Secret key for embedding
    pub secret_key: Option<String>,
    /// Vocabulary partition size
    pub vocab_partition: f32,
    /// Minimum text length for embedding
    pub min_length: usize,
    /// Enable adaptive strength
    pub adaptive: bool,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            watermark_type: WatermarkType::Soft,
            strength: 0.5,
            secret_key: None,
            vocab_partition: 0.5,
            min_length: 10,
            adaptive: false,
        }
    }
}

impl WatermarkConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set watermark type
    pub fn watermark_type(mut self, t: WatermarkType) -> Self {
        self.watermark_type = t;
        self
    }

    /// Set strength
    pub fn strength(mut self, s: f32) -> Self {
        self.strength = s.clamp(0.0, 1.0);
        self
    }

    /// Set secret key
    pub fn secret_key(mut self, key: impl Into<String>) -> Self {
        self.secret_key = Some(key.into());
        self
    }

    /// Set vocabulary partition
    pub fn vocab_partition(mut self, p: f32) -> Self {
        self.vocab_partition = p.clamp(0.1, 0.9);
        self
    }

    /// Set minimum length
    pub fn min_length(mut self, n: usize) -> Self {
        self.min_length = n;
        self
    }

    /// Enable adaptive strength
    pub fn adaptive(mut self, enabled: bool) -> Self {
        self.adaptive = enabled;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.watermark_type.requires_key() && self.secret_key.is_none() {
            return Err("Secret key required for this watermark type".to_string());
        }
        if self.min_length == 0 {
            return Err("Minimum length must be > 0".to_string());
        }
        Ok(())
    }
}

/// Detection configuration
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Detection method
    pub method: DetectionMethod,
    /// Detection threshold
    pub threshold: f32,
    /// Window size for detection
    pub window_size: usize,
    /// Secret key for verification
    pub secret_key: Option<String>,
    /// Minimum confidence to report
    pub min_confidence: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            method: DetectionMethod::Statistical,
            threshold: 0.7,
            window_size: 50,
            secret_key: None,
            min_confidence: 0.5,
        }
    }
}

impl DetectionConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set method
    pub fn method(mut self, m: DetectionMethod) -> Self {
        self.method = m;
        self
    }

    /// Set threshold
    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Set window size
    pub fn window_size(mut self, n: usize) -> Self {
        self.window_size = n;
        self
    }

    /// Set secret key
    pub fn secret_key(mut self, key: impl Into<String>) -> Self {
        self.secret_key = Some(key.into());
        self
    }

    /// Set min confidence
    pub fn min_confidence(mut self, c: f32) -> Self {
        self.min_confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("Window size must be > 0".to_string());
        }
        if self.method == DetectionMethod::KeyBased && self.secret_key.is_none() {
            return Err("Secret key required for key-based detection".to_string());
        }
        Ok(())
    }
}

/// Watermark embedding result
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// Watermarked text/tokens
    pub output: String,
    /// Tokens modified
    pub tokens_modified: usize,
    /// Total tokens
    pub total_tokens: usize,
    /// Effective strength
    pub effective_strength: f32,
    /// Success flag
    pub success: bool,
}

impl EmbedResult {
    /// Create success result
    pub fn success(output: String, tokens_modified: usize, total_tokens: usize) -> Self {
        Self {
            output,
            tokens_modified,
            total_tokens,
            effective_strength: if total_tokens > 0 {
                tokens_modified as f32 / total_tokens as f32
            } else {
                0.0
            },
            success: true,
        }
    }

    /// Create failure result
    pub fn failure(reason: &str) -> Self {
        Self {
            output: reason.to_string(),
            tokens_modified: 0,
            total_tokens: 0,
            effective_strength: 0.0,
            success: false,
        }
    }

    /// Get modification rate
    pub fn modification_rate(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.tokens_modified as f32 / self.total_tokens as f32
    }
}

/// Watermark detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether watermark detected
    pub detected: bool,
    /// Confidence score
    pub confidence: f32,
    /// Detected watermark type
    pub watermark_type: Option<WatermarkType>,
    /// Z-score (for statistical detection)
    pub z_score: Option<f32>,
    /// P-value (for statistical detection)
    pub p_value: Option<f32>,
    /// Token count analyzed
    pub tokens_analyzed: usize,
}

impl DetectionResult {
    /// Create positive detection
    pub fn detected(confidence: f32, watermark_type: WatermarkType) -> Self {
        Self {
            detected: true,
            confidence: confidence.clamp(0.0, 1.0),
            watermark_type: Some(watermark_type),
            z_score: None,
            p_value: None,
            tokens_analyzed: 0,
        }
    }

    /// Create negative detection
    pub fn not_detected(tokens_analyzed: usize) -> Self {
        Self {
            detected: false,
            confidence: 0.0,
            watermark_type: None,
            z_score: None,
            p_value: None,
            tokens_analyzed,
        }
    }

    /// Set z-score
    pub fn with_z_score(mut self, z: f32) -> Self {
        self.z_score = Some(z);
        self
    }

    /// Set p-value
    pub fn with_p_value(mut self, p: f32) -> Self {
        self.p_value = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set tokens analyzed
    pub fn with_tokens(mut self, n: usize) -> Self {
        self.tokens_analyzed = n;
        self
    }

    /// Check if significant (p < 0.05)
    pub fn is_significant(&self) -> bool {
        self.p_value.map(|p| p < 0.05).unwrap_or(false)
    }
}

/// Watermarking statistics
#[derive(Debug, Clone, Default)]
pub struct WatermarkStats {
    /// Total texts processed
    pub texts_processed: usize,
    /// Successful embeddings
    pub embeddings_success: usize,
    /// Successful detections
    pub detections_positive: usize,
    /// Total tokens modified
    pub tokens_modified: usize,
    /// Average strength
    pub avg_strength: f32,
}

impl WatermarkStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record embedding
    pub fn record_embed(&mut self, result: &EmbedResult) {
        self.texts_processed += 1;
        if result.success {
            self.embeddings_success += 1;
            self.tokens_modified += result.tokens_modified;
            let n = self.embeddings_success as f32;
            self.avg_strength = (self.avg_strength * (n - 1.0) + result.effective_strength) / n;
        }
    }

    /// Record detection
    pub fn record_detect(&mut self, result: &DetectionResult) {
        if result.detected {
            self.detections_positive += 1;
        }
    }

    /// Get embedding success rate
    pub fn embed_success_rate(&self) -> f64 {
        if self.texts_processed == 0 {
            return 0.0;
        }
        self.embeddings_success as f64 / self.texts_processed as f64
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Watermark: {} texts, {:.1}% embed success, {:.2} avg strength",
            self.texts_processed,
            self.embed_success_rate() * 100.0,
            self.avg_strength
        )
    }
}

/// Vocabulary bias configuration for soft watermarks
#[derive(Debug, Clone)]
pub struct VocabBiasConfig {
    /// Green list fraction (tokens to boost)
    pub green_fraction: f32,
    /// Bias delta (logit adjustment)
    pub bias_delta: f32,
    /// Context window for hashing
    pub context_window: usize,
    /// Hash seed
    pub seed: u64,
}

impl Default for VocabBiasConfig {
    fn default() -> Self {
        Self {
            green_fraction: 0.5,
            bias_delta: 2.0,
            context_window: 1,
            seed: 42,
        }
    }
}

impl VocabBiasConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set green fraction
    pub fn green_fraction(mut self, f: f32) -> Self {
        self.green_fraction = f.clamp(0.1, 0.9);
        self
    }

    /// Set bias delta
    pub fn bias_delta(mut self, d: f32) -> Self {
        self.bias_delta = d.clamp(0.0, 10.0);
        self
    }

    /// Set context window
    pub fn context_window(mut self, n: usize) -> Self {
        self.context_window = n.max(1);
        self
    }

    /// Set seed
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Calculate expected green ratio
    ///
    /// When green tokens are biased by `bias_delta`, the expected ratio
    /// of green tokens in output is:
    /// γ * exp(δ) / (γ * exp(δ) + (1-γ))
    /// where γ = green_fraction, δ = bias_delta
    pub fn expected_green_ratio(&self) -> f32 {
        let gamma = self.green_fraction;
        let exp_delta = self.bias_delta.exp();
        gamma * exp_delta / (gamma * exp_delta + (1.0 - gamma))
    }
}

/// Calculate z-score for watermark detection
pub fn calculate_z_score(green_count: usize, total_count: usize, expected_ratio: f32) -> f32 {
    if total_count == 0 {
        return 0.0;
    }

    let observed = green_count as f32 / total_count as f32;
    let expected = expected_ratio;
    let std_dev = (expected * (1.0 - expected) / total_count as f32).sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    (observed - expected) / std_dev
}

/// Calculate p-value from z-score (one-tailed)
pub fn z_score_to_p_value(z: f32) -> f32 {
    // Approximation using error function
    let x = z / (2.0_f32).sqrt();
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;

    let erf = 1.0 - (a1 * t + a2 * t * t + a3 * t.powi(3) + a4 * t.powi(4) + a5 * t.powi(5))
        * (-x * x).exp();

    let erf = if x < 0.0 { -erf } else { erf };
    (1.0 - erf) / 2.0
}

/// Estimate entropy reduction from watermarking
pub fn estimate_entropy_reduction(strength: f32, vocab_size: usize) -> f32 {
    // Simplified model: watermarking reduces effective vocab
    let effective_vocab = vocab_size as f32 * (1.0 - strength * 0.5);
    let original_entropy = (vocab_size as f32).ln();
    let reduced_entropy = effective_vocab.ln();
    original_entropy - reduced_entropy
}

/// Simple hash for vocabulary partitioning
pub fn hash_token_context(token_id: u64, context: &[u64], seed: u64) -> bool {
    let mut hash = seed;
    hash = hash.wrapping_mul(0x517cc1b727220a95);
    hash = hash.wrapping_add(token_id);

    for &ctx in context {
        hash = hash.wrapping_mul(0x517cc1b727220a95);
        hash = hash.wrapping_add(ctx);
    }

    // Return true for "green" list
    hash % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_type_default() {
        assert_eq!(WatermarkType::default(), WatermarkType::Soft);
    }

    #[test]
    fn test_watermark_type_as_str() {
        assert_eq!(WatermarkType::Soft.as_str(), "soft");
        assert_eq!(WatermarkType::Hard.as_str(), "hard");
        assert_eq!(WatermarkType::Semantic.as_str(), "semantic");
        assert_eq!(WatermarkType::Statistical.as_str(), "statistical");
        assert_eq!(WatermarkType::Neural.as_str(), "neural");
        assert_eq!(WatermarkType::Cryptographic.as_str(), "cryptographic");
    }

    #[test]
    fn test_watermark_type_parse() {
        assert_eq!(WatermarkType::parse("soft"), Some(WatermarkType::Soft));
        assert_eq!(WatermarkType::parse("bias"), Some(WatermarkType::Soft));
        assert_eq!(WatermarkType::parse("deterministic"), Some(WatermarkType::Hard));
        assert_eq!(WatermarkType::parse("meaning"), Some(WatermarkType::Semantic));
        assert_eq!(WatermarkType::parse("stats"), Some(WatermarkType::Statistical));
        assert_eq!(WatermarkType::parse("nn"), Some(WatermarkType::Neural));
        assert_eq!(WatermarkType::parse("crypto"), Some(WatermarkType::Cryptographic));
        assert_eq!(WatermarkType::parse("unknown"), None);
    }

    #[test]
    fn test_watermark_type_list_all() {
        assert_eq!(WatermarkType::list_all().len(), 6);
    }

    #[test]
    fn test_watermark_type_requires_key() {
        assert!(!WatermarkType::Soft.requires_key());
        assert!(!WatermarkType::Hard.requires_key());
        assert!(WatermarkType::Neural.requires_key());
        assert!(WatermarkType::Cryptographic.requires_key());
    }

    #[test]
    fn test_watermark_type_robustness() {
        assert_eq!(WatermarkType::Soft.robustness(), 4);
        assert_eq!(WatermarkType::Cryptographic.robustness(), 9);
    }

    #[test]
    fn test_detection_method_as_str() {
        assert_eq!(DetectionMethod::Statistical.as_str(), "statistical");
        assert_eq!(DetectionMethod::KeyBased.as_str(), "key_based");
        assert_eq!(DetectionMethod::PatternMatch.as_str(), "pattern_match");
        assert_eq!(DetectionMethod::NeuralNet.as_str(), "neural_net");
        assert_eq!(DetectionMethod::Hybrid.as_str(), "hybrid");
    }

    #[test]
    fn test_detection_method_parse() {
        assert_eq!(DetectionMethod::parse("stats"), Some(DetectionMethod::Statistical));
        assert_eq!(DetectionMethod::parse("key"), Some(DetectionMethod::KeyBased));
        assert_eq!(DetectionMethod::parse("pattern"), Some(DetectionMethod::PatternMatch));
        assert_eq!(DetectionMethod::parse("neural"), Some(DetectionMethod::NeuralNet));
        assert_eq!(DetectionMethod::parse("combined"), Some(DetectionMethod::Hybrid));
        assert_eq!(DetectionMethod::parse("unknown"), None);
    }

    #[test]
    fn test_detection_method_requires_model() {
        assert!(!DetectionMethod::Statistical.requires_model());
        assert!(!DetectionMethod::KeyBased.requires_model());
        assert!(DetectionMethod::NeuralNet.requires_model());
        assert!(DetectionMethod::Hybrid.requires_model());
    }

    #[test]
    fn test_watermark_config_default() {
        let config = WatermarkConfig::default();
        assert_eq!(config.watermark_type, WatermarkType::Soft);
        assert_eq!(config.strength, 0.5);
        assert!(config.secret_key.is_none());
    }

    #[test]
    fn test_watermark_config_builder() {
        let config = WatermarkConfig::new()
            .watermark_type(WatermarkType::Hard)
            .strength(0.8)
            .secret_key("my_secret")
            .vocab_partition(0.6)
            .min_length(20)
            .adaptive(true);

        assert_eq!(config.watermark_type, WatermarkType::Hard);
        assert_eq!(config.strength, 0.8);
        assert_eq!(config.secret_key, Some("my_secret".to_string()));
        assert_eq!(config.vocab_partition, 0.6);
        assert_eq!(config.min_length, 20);
        assert!(config.adaptive);
    }

    #[test]
    fn test_watermark_config_strength_clamp() {
        let config = WatermarkConfig::new().strength(1.5);
        assert_eq!(config.strength, 1.0);

        let config = WatermarkConfig::new().strength(-0.5);
        assert_eq!(config.strength, 0.0);
    }

    #[test]
    fn test_watermark_config_validate() {
        let valid = WatermarkConfig::default();
        assert!(valid.validate().is_ok());

        let crypto_no_key = WatermarkConfig::new()
            .watermark_type(WatermarkType::Cryptographic);
        assert!(crypto_no_key.validate().is_err());

        let zero_length = WatermarkConfig::new().min_length(0);
        assert!(zero_length.validate().is_err());
    }

    #[test]
    fn test_detection_config_default() {
        let config = DetectionConfig::default();
        assert_eq!(config.method, DetectionMethod::Statistical);
        assert_eq!(config.threshold, 0.7);
        assert_eq!(config.window_size, 50);
    }

    #[test]
    fn test_detection_config_builder() {
        let config = DetectionConfig::new()
            .method(DetectionMethod::KeyBased)
            .threshold(0.9)
            .window_size(100)
            .secret_key("detect_key")
            .min_confidence(0.6);

        assert_eq!(config.method, DetectionMethod::KeyBased);
        assert_eq!(config.threshold, 0.9);
        assert_eq!(config.window_size, 100);
        assert!(config.secret_key.is_some());
        assert_eq!(config.min_confidence, 0.6);
    }

    #[test]
    fn test_detection_config_validate() {
        let valid = DetectionConfig::default();
        assert!(valid.validate().is_ok());

        let zero_window = DetectionConfig::new().window_size(0);
        assert!(zero_window.validate().is_err());

        let key_based_no_key = DetectionConfig::new()
            .method(DetectionMethod::KeyBased);
        assert!(key_based_no_key.validate().is_err());
    }

    #[test]
    fn test_embed_result_success() {
        let result = EmbedResult::success("watermarked text".to_string(), 5, 10);
        assert!(result.success);
        assert_eq!(result.tokens_modified, 5);
        assert!((result.modification_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_embed_result_failure() {
        let result = EmbedResult::failure("text too short");
        assert!(!result.success);
        assert_eq!(result.modification_rate(), 0.0);
    }

    #[test]
    fn test_detection_result_detected() {
        let result = DetectionResult::detected(0.95, WatermarkType::Soft)
            .with_z_score(4.5)
            .with_p_value(0.001)
            .with_tokens(100);

        assert!(result.detected);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.watermark_type, Some(WatermarkType::Soft));
        assert!(result.is_significant());
        assert_eq!(result.tokens_analyzed, 100);
    }

    #[test]
    fn test_detection_result_not_detected() {
        let result = DetectionResult::not_detected(50);
        assert!(!result.detected);
        assert_eq!(result.confidence, 0.0);
        assert!(result.watermark_type.is_none());
        assert!(!result.is_significant());
    }

    #[test]
    fn test_watermark_stats_new() {
        let stats = WatermarkStats::new();
        assert_eq!(stats.texts_processed, 0);
        assert_eq!(stats.embed_success_rate(), 0.0);
    }

    #[test]
    fn test_watermark_stats_record_embed() {
        let mut stats = WatermarkStats::new();
        stats.record_embed(&EmbedResult::success("text".to_string(), 5, 10));
        stats.record_embed(&EmbedResult::failure("error"));

        assert_eq!(stats.texts_processed, 2);
        assert_eq!(stats.embeddings_success, 1);
        assert_eq!(stats.tokens_modified, 5);
        assert!((stats.embed_success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_watermark_stats_record_detect() {
        let mut stats = WatermarkStats::new();
        stats.record_detect(&DetectionResult::detected(0.9, WatermarkType::Soft));
        stats.record_detect(&DetectionResult::not_detected(50));

        assert_eq!(stats.detections_positive, 1);
    }

    #[test]
    fn test_watermark_stats_format() {
        let mut stats = WatermarkStats::new();
        stats.record_embed(&EmbedResult::success("text".to_string(), 5, 10));
        let formatted = stats.format();

        assert!(formatted.contains("1 texts"));
        assert!(formatted.contains("100.0% embed success"));
    }

    #[test]
    fn test_vocab_bias_config_default() {
        let config = VocabBiasConfig::default();
        assert_eq!(config.green_fraction, 0.5);
        assert_eq!(config.bias_delta, 2.0);
        assert_eq!(config.context_window, 1);
    }

    #[test]
    fn test_vocab_bias_config_builder() {
        let config = VocabBiasConfig::new()
            .green_fraction(0.4)
            .bias_delta(3.0)
            .context_window(2)
            .seed(123);

        assert_eq!(config.green_fraction, 0.4);
        assert_eq!(config.bias_delta, 3.0);
        assert_eq!(config.context_window, 2);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn test_vocab_bias_expected_green_ratio() {
        let config = VocabBiasConfig::new()
            .green_fraction(0.5)
            .bias_delta(2.0);

        let ratio = config.expected_green_ratio();
        assert!(ratio > 0.5); // Should be biased toward green
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_calculate_z_score() {
        let z = calculate_z_score(70, 100, 0.5);
        assert!(z > 0.0); // 70% > 50% expected

        let z = calculate_z_score(30, 100, 0.5);
        assert!(z < 0.0); // 30% < 50% expected

        let z = calculate_z_score(0, 0, 0.5);
        assert_eq!(z, 0.0); // Edge case
    }

    #[test]
    fn test_z_score_to_p_value() {
        let p = z_score_to_p_value(0.0);
        assert!((p - 0.5).abs() < 0.1); // z=0 should give p~0.5

        let p = z_score_to_p_value(2.0);
        assert!(p < 0.05); // z=2 should give low p-value

        let p = z_score_to_p_value(-2.0);
        assert!(p > 0.9); // z=-2 should give high p-value (one-tailed)
    }

    #[test]
    fn test_estimate_entropy_reduction() {
        let reduction = estimate_entropy_reduction(0.0, 10000);
        assert!(reduction < 0.1); // No watermark = minimal reduction

        let reduction = estimate_entropy_reduction(0.5, 10000);
        assert!(reduction > 0.0); // Medium watermark = some reduction

        let reduction_strong = estimate_entropy_reduction(0.8, 10000);
        let reduction_weak = estimate_entropy_reduction(0.2, 10000);
        assert!(reduction_strong > reduction_weak);
    }

    #[test]
    fn test_hash_token_context() {
        let result1 = hash_token_context(1, &[0], 42);
        let result2 = hash_token_context(2, &[0], 42);

        // Different tokens should potentially give different results
        // (not always, but deterministic)
        let _ = result1;
        let _ = result2;

        // Same inputs = same output
        assert_eq!(
            hash_token_context(100, &[1, 2, 3], 999),
            hash_token_context(100, &[1, 2, 3], 999)
        );
    }

    #[test]
    fn test_hash_token_context_different_seeds() {
        // Different seeds may give different results
        let r1 = hash_token_context(1, &[], 42);
        let r2 = hash_token_context(1, &[], 43);

        // Results may differ (not guaranteed, but likely)
        let _ = (r1, r2);
    }
}
