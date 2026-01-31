//! Content Guardrails
//!
//! Safety guardrails for content filtering and moderation.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.safety import GuardrailType, ContentFilter, create_guardrail
//! filter = create_guardrail(GuardrailType.CONTENT_FILTER, threshold=0.8)
//! ```

use std::collections::HashSet;

/// Guardrail type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardrailType {
    /// Content filtering (default)
    #[default]
    ContentFilter,
    /// Input validation
    InputValidation,
    /// Output sanitization
    OutputSanitization,
    /// Rate limiting
    RateLimiting,
    /// Topic restriction
    TopicRestriction,
    /// Format enforcement
    FormatEnforcement,
}

impl GuardrailType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ContentFilter => "content_filter",
            Self::InputValidation => "input_validation",
            Self::OutputSanitization => "output_sanitization",
            Self::RateLimiting => "rate_limiting",
            Self::TopicRestriction => "topic_restriction",
            Self::FormatEnforcement => "format_enforcement",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "content_filter" | "filter" => Some(Self::ContentFilter),
            "input_validation" | "input" => Some(Self::InputValidation),
            "output_sanitization" | "output" => Some(Self::OutputSanitization),
            "rate_limiting" | "rate" => Some(Self::RateLimiting),
            "topic_restriction" | "topic" => Some(Self::TopicRestriction),
            "format_enforcement" | "format" => Some(Self::FormatEnforcement),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::ContentFilter,
            Self::InputValidation,
            Self::OutputSanitization,
            Self::RateLimiting,
            Self::TopicRestriction,
            Self::FormatEnforcement,
        ]
    }
}

/// Content category for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ContentCategory {
    /// Safe content (default)
    #[default]
    Safe,
    /// Harmful content
    Harmful,
    /// Hateful content
    Hateful,
    /// Sexual content
    Sexual,
    /// Violent content
    Violent,
    /// Self-harm content
    SelfHarm,
    /// Illegal content
    Illegal,
    /// Misinformation
    Misinformation,
}

impl ContentCategory {
    /// Get category name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Harmful => "harmful",
            Self::Hateful => "hateful",
            Self::Sexual => "sexual",
            Self::Violent => "violent",
            Self::SelfHarm => "self_harm",
            Self::Illegal => "illegal",
            Self::Misinformation => "misinformation",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "safe" => Some(Self::Safe),
            "harmful" | "harm" => Some(Self::Harmful),
            "hateful" | "hate" => Some(Self::Hateful),
            "sexual" | "nsfw" => Some(Self::Sexual),
            "violent" | "violence" => Some(Self::Violent),
            "self_harm" | "selfharm" => Some(Self::SelfHarm),
            "illegal" => Some(Self::Illegal),
            "misinformation" | "misinfo" => Some(Self::Misinformation),
            _ => None,
        }
    }

    /// Check if category is harmful
    pub fn is_harmful(&self) -> bool {
        !matches!(self, Self::Safe)
    }

    /// Get severity level (0-10)
    pub fn severity(&self) -> u8 {
        match self {
            Self::Safe => 0,
            Self::Misinformation => 3,
            Self::Sexual => 5,
            Self::Hateful => 7,
            Self::Violent => 8,
            Self::Harmful => 8,
            Self::SelfHarm => 9,
            Self::Illegal => 10,
        }
    }
}

/// Guardrail action when triggered
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardrailAction {
    /// Allow content (default)
    #[default]
    Allow,
    /// Block content
    Block,
    /// Warn user
    Warn,
    /// Flag for review
    Flag,
    /// Redact content
    Redact,
    /// Replace content
    Replace,
}

impl GuardrailAction {
    /// Get action name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block => "block",
            Self::Warn => "warn",
            Self::Flag => "flag",
            Self::Redact => "redact",
            Self::Replace => "replace",
        }
    }

    /// Check if action blocks content
    pub fn blocks_content(&self) -> bool {
        matches!(self, Self::Block | Self::Redact | Self::Replace)
    }
}

/// Content filter configuration
#[derive(Debug, Clone)]
pub struct ContentFilterConfig {
    /// Filter threshold (0.0-1.0)
    pub threshold: f32,
    /// Categories to filter
    pub categories: HashSet<ContentCategory>,
    /// Action on violation
    pub action: GuardrailAction,
    /// Enable strict mode
    pub strict_mode: bool,
    /// Replacement text for redaction
    pub replacement_text: String,
}

impl Default for ContentFilterConfig {
    fn default() -> Self {
        let mut categories = HashSet::new();
        categories.insert(ContentCategory::Harmful);
        categories.insert(ContentCategory::Hateful);
        categories.insert(ContentCategory::Illegal);

        Self {
            threshold: 0.8,
            categories,
            action: GuardrailAction::Block,
            strict_mode: false,
            replacement_text: "[REDACTED]".to_string(),
        }
    }
}

impl ContentFilterConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set threshold
    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Add category
    pub fn category(mut self, c: ContentCategory) -> Self {
        self.categories.insert(c);
        self
    }

    /// Set categories
    pub fn categories(mut self, cats: HashSet<ContentCategory>) -> Self {
        self.categories = cats;
        self
    }

    /// Set action
    pub fn action(mut self, a: GuardrailAction) -> Self {
        self.action = a;
        self
    }

    /// Enable strict mode
    pub fn strict_mode(mut self, enabled: bool) -> Self {
        self.strict_mode = enabled;
        self
    }

    /// Set replacement text
    pub fn replacement_text(mut self, text: impl Into<String>) -> Self {
        self.replacement_text = text.into();
        self
    }

    /// Check if category is filtered
    pub fn filters_category(&self, cat: ContentCategory) -> bool {
        self.categories.contains(&cat)
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.threshold < 0.0 || self.threshold > 1.0 {
            return Err("Threshold must be between 0.0 and 1.0".to_string());
        }
        if self.categories.is_empty() {
            return Err("Must specify at least one category to filter".to_string());
        }
        Ok(())
    }
}

/// Input validation configuration
#[derive(Debug, Clone)]
pub struct InputValidationConfig {
    /// Maximum input length
    pub max_length: usize,
    /// Minimum input length
    pub min_length: usize,
    /// Allowed characters pattern
    pub allowed_chars: Option<String>,
    /// Blocked patterns
    pub blocked_patterns: Vec<String>,
    /// Strip whitespace
    pub strip_whitespace: bool,
}

impl Default for InputValidationConfig {
    fn default() -> Self {
        Self {
            max_length: 10000,
            min_length: 1,
            allowed_chars: None,
            blocked_patterns: Vec::new(),
            strip_whitespace: true,
        }
    }
}

impl InputValidationConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max length
    pub fn max_length(mut self, n: usize) -> Self {
        self.max_length = n;
        self
    }

    /// Set min length
    pub fn min_length(mut self, n: usize) -> Self {
        self.min_length = n;
        self
    }

    /// Set allowed chars
    pub fn allowed_chars(mut self, pattern: impl Into<String>) -> Self {
        self.allowed_chars = Some(pattern.into());
        self
    }

    /// Add blocked pattern
    pub fn block_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.blocked_patterns.push(pattern.into());
        self
    }

    /// Set strip whitespace
    pub fn strip_whitespace(mut self, enabled: bool) -> Self {
        self.strip_whitespace = enabled;
        self
    }

    /// Validate input
    pub fn validate_input(&self, input: &str) -> Result<(), String> {
        let text = if self.strip_whitespace {
            input.trim()
        } else {
            input
        };

        if text.len() < self.min_length {
            return Err(format!(
                "Input too short: {} < {}",
                text.len(),
                self.min_length
            ));
        }

        if text.len() > self.max_length {
            return Err(format!(
                "Input too long: {} > {}",
                text.len(),
                self.max_length
            ));
        }

        for pattern in &self.blocked_patterns {
            if text.contains(pattern) {
                return Err(format!("Input contains blocked pattern: {}", pattern));
            }
        }

        Ok(())
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Max requests per window
    pub max_requests: usize,
    /// Window size in seconds
    pub window_seconds: u64,
    /// Cooldown on limit hit (seconds)
    pub cooldown_seconds: u64,
    /// Enable burst allowance
    pub allow_burst: bool,
    /// Burst size
    pub burst_size: usize,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_seconds: 60,
            cooldown_seconds: 30,
            allow_burst: false,
            burst_size: 10,
        }
    }
}

impl RateLimitConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max requests
    pub fn max_requests(mut self, n: usize) -> Self {
        self.max_requests = n;
        self
    }

    /// Set window size
    pub fn window_seconds(mut self, s: u64) -> Self {
        self.window_seconds = s;
        self
    }

    /// Set cooldown
    pub fn cooldown_seconds(mut self, s: u64) -> Self {
        self.cooldown_seconds = s;
        self
    }

    /// Enable burst
    pub fn allow_burst(mut self, enabled: bool) -> Self {
        self.allow_burst = enabled;
        self
    }

    /// Set burst size
    pub fn burst_size(mut self, n: usize) -> Self {
        self.burst_size = n;
        self
    }

    /// Calculate requests per second
    pub fn requests_per_second(&self) -> f64 {
        if self.window_seconds == 0 {
            return 0.0;
        }
        self.max_requests as f64 / self.window_seconds as f64
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.max_requests == 0 {
            return Err("Max requests must be > 0".to_string());
        }
        if self.window_seconds == 0 {
            return Err("Window size must be > 0".to_string());
        }
        if self.allow_burst && self.burst_size == 0 {
            return Err("Burst size must be > 0 when burst is enabled".to_string());
        }
        Ok(())
    }
}

/// Guardrail check result
#[derive(Debug, Clone)]
pub struct GuardrailResult {
    /// Whether check passed
    pub passed: bool,
    /// Action taken
    pub action: GuardrailAction,
    /// Detected categories
    pub categories: Vec<ContentCategory>,
    /// Confidence scores
    pub scores: Vec<f32>,
    /// Reason for action
    pub reason: Option<String>,
    /// Modified content (if redacted/replaced)
    pub modified_content: Option<String>,
}

impl GuardrailResult {
    /// Create passed result
    pub fn pass() -> Self {
        Self {
            passed: true,
            action: GuardrailAction::Allow,
            categories: vec![ContentCategory::Safe],
            scores: vec![1.0],
            reason: None,
            modified_content: None,
        }
    }

    /// Create blocked result
    pub fn block(category: ContentCategory, score: f32) -> Self {
        Self {
            passed: false,
            action: GuardrailAction::Block,
            categories: vec![category],
            scores: vec![score],
            reason: Some(format!("Content blocked: {}", category.as_str())),
            modified_content: None,
        }
    }

    /// Create warned result
    pub fn warn(category: ContentCategory, score: f32) -> Self {
        Self {
            passed: true,
            action: GuardrailAction::Warn,
            categories: vec![category],
            scores: vec![score],
            reason: Some(format!("Content warning: {}", category.as_str())),
            modified_content: None,
        }
    }

    /// Create redacted result
    pub fn redact(category: ContentCategory, score: f32, redacted: String) -> Self {
        Self {
            passed: true,
            action: GuardrailAction::Redact,
            categories: vec![category],
            scores: vec![score],
            reason: Some(format!("Content redacted: {}", category.as_str())),
            modified_content: Some(redacted),
        }
    }

    /// Add category
    pub fn with_category(mut self, cat: ContentCategory, score: f32) -> Self {
        self.categories.push(cat);
        self.scores.push(score);
        self
    }

    /// Set reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Get highest score
    pub fn max_score(&self) -> f32 {
        self.scores.iter().copied().fold(0.0, f32::max)
    }

    /// Get most severe category
    pub fn most_severe_category(&self) -> ContentCategory {
        self.categories
            .iter()
            .max_by_key(|c| c.severity())
            .copied()
            .unwrap_or(ContentCategory::Safe)
    }
}

/// Guardrail statistics
#[derive(Debug, Clone, Default)]
pub struct GuardrailStats {
    /// Total checks
    pub total_checks: usize,
    /// Passed checks
    pub passed_checks: usize,
    /// Blocked checks
    pub blocked_checks: usize,
    /// Warned checks
    pub warned_checks: usize,
    /// Category counts
    pub category_counts: std::collections::HashMap<String, usize>,
}

impl GuardrailStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record check
    pub fn record(&mut self, result: &GuardrailResult) {
        self.total_checks += 1;
        if result.passed {
            self.passed_checks += 1;
        } else {
            self.blocked_checks += 1;
        }
        if result.action == GuardrailAction::Warn {
            self.warned_checks += 1;
        }
        for cat in &result.categories {
            *self
                .category_counts
                .entry(cat.as_str().to_string())
                .or_insert(0) += 1;
        }
    }

    /// Get pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 1.0;
        }
        self.passed_checks as f64 / self.total_checks as f64
    }

    /// Get block rate
    pub fn block_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.blocked_checks as f64 / self.total_checks as f64
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Guardrails: {} checks ({:.1}% passed, {:.1}% blocked)",
            self.total_checks,
            self.pass_rate() * 100.0,
            self.block_rate() * 100.0
        )
    }
}

/// Calculate content safety score (0.0-1.0, higher is safer)
pub fn calculate_safety_score(categories: &[ContentCategory], scores: &[f32]) -> f32 {
    if categories.is_empty() || scores.is_empty() {
        return 1.0;
    }

    let weighted_sum: f32 = categories
        .iter()
        .zip(scores.iter())
        .map(|(cat, score)| cat.severity() as f32 * score / 10.0)
        .sum();

    let max_weight = categories.len() as f32;
    let danger_score = weighted_sum / max_weight;

    (1.0 - danger_score).clamp(0.0, 1.0)
}

/// Estimate moderation latency in ms
pub fn estimate_moderation_latency(text_length: usize, categories: usize) -> u64 {
    let base_latency = 10;
    let text_factor = text_length / 1000;
    let category_factor = categories * 5;
    (base_latency + text_factor + category_factor) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrail_type_default() {
        assert_eq!(GuardrailType::default(), GuardrailType::ContentFilter);
    }

    #[test]
    fn test_guardrail_type_as_str() {
        assert_eq!(GuardrailType::ContentFilter.as_str(), "content_filter");
        assert_eq!(GuardrailType::InputValidation.as_str(), "input_validation");
        assert_eq!(GuardrailType::OutputSanitization.as_str(), "output_sanitization");
        assert_eq!(GuardrailType::RateLimiting.as_str(), "rate_limiting");
        assert_eq!(GuardrailType::TopicRestriction.as_str(), "topic_restriction");
        assert_eq!(GuardrailType::FormatEnforcement.as_str(), "format_enforcement");
    }

    #[test]
    fn test_guardrail_type_parse() {
        assert_eq!(GuardrailType::parse("filter"), Some(GuardrailType::ContentFilter));
        assert_eq!(GuardrailType::parse("input"), Some(GuardrailType::InputValidation));
        assert_eq!(GuardrailType::parse("output"), Some(GuardrailType::OutputSanitization));
        assert_eq!(GuardrailType::parse("rate"), Some(GuardrailType::RateLimiting));
        assert_eq!(GuardrailType::parse("topic"), Some(GuardrailType::TopicRestriction));
        assert_eq!(GuardrailType::parse("format"), Some(GuardrailType::FormatEnforcement));
        assert_eq!(GuardrailType::parse("unknown"), None);
    }

    #[test]
    fn test_guardrail_type_list_all() {
        assert_eq!(GuardrailType::list_all().len(), 6);
    }

    #[test]
    fn test_content_category_default() {
        assert_eq!(ContentCategory::default(), ContentCategory::Safe);
    }

    #[test]
    fn test_content_category_as_str() {
        assert_eq!(ContentCategory::Safe.as_str(), "safe");
        assert_eq!(ContentCategory::Harmful.as_str(), "harmful");
        assert_eq!(ContentCategory::Hateful.as_str(), "hateful");
        assert_eq!(ContentCategory::Sexual.as_str(), "sexual");
        assert_eq!(ContentCategory::Violent.as_str(), "violent");
        assert_eq!(ContentCategory::SelfHarm.as_str(), "self_harm");
        assert_eq!(ContentCategory::Illegal.as_str(), "illegal");
        assert_eq!(ContentCategory::Misinformation.as_str(), "misinformation");
    }

    #[test]
    fn test_content_category_parse() {
        assert_eq!(ContentCategory::parse("safe"), Some(ContentCategory::Safe));
        assert_eq!(ContentCategory::parse("harm"), Some(ContentCategory::Harmful));
        assert_eq!(ContentCategory::parse("hate"), Some(ContentCategory::Hateful));
        assert_eq!(ContentCategory::parse("nsfw"), Some(ContentCategory::Sexual));
        assert_eq!(ContentCategory::parse("violence"), Some(ContentCategory::Violent));
        assert_eq!(ContentCategory::parse("selfharm"), Some(ContentCategory::SelfHarm));
        assert_eq!(ContentCategory::parse("illegal"), Some(ContentCategory::Illegal));
        assert_eq!(ContentCategory::parse("misinfo"), Some(ContentCategory::Misinformation));
        assert_eq!(ContentCategory::parse("unknown"), None);
    }

    #[test]
    fn test_content_category_is_harmful() {
        assert!(!ContentCategory::Safe.is_harmful());
        assert!(ContentCategory::Harmful.is_harmful());
        assert!(ContentCategory::Hateful.is_harmful());
        assert!(ContentCategory::Illegal.is_harmful());
    }

    #[test]
    fn test_content_category_severity() {
        assert_eq!(ContentCategory::Safe.severity(), 0);
        assert_eq!(ContentCategory::Misinformation.severity(), 3);
        assert_eq!(ContentCategory::Sexual.severity(), 5);
        assert_eq!(ContentCategory::Illegal.severity(), 10);
    }

    #[test]
    fn test_guardrail_action_as_str() {
        assert_eq!(GuardrailAction::Allow.as_str(), "allow");
        assert_eq!(GuardrailAction::Block.as_str(), "block");
        assert_eq!(GuardrailAction::Warn.as_str(), "warn");
        assert_eq!(GuardrailAction::Flag.as_str(), "flag");
        assert_eq!(GuardrailAction::Redact.as_str(), "redact");
        assert_eq!(GuardrailAction::Replace.as_str(), "replace");
    }

    #[test]
    fn test_guardrail_action_blocks_content() {
        assert!(!GuardrailAction::Allow.blocks_content());
        assert!(GuardrailAction::Block.blocks_content());
        assert!(!GuardrailAction::Warn.blocks_content());
        assert!(!GuardrailAction::Flag.blocks_content());
        assert!(GuardrailAction::Redact.blocks_content());
        assert!(GuardrailAction::Replace.blocks_content());
    }

    #[test]
    fn test_content_filter_config_default() {
        let config = ContentFilterConfig::default();
        assert_eq!(config.threshold, 0.8);
        assert!(config.categories.contains(&ContentCategory::Harmful));
        assert_eq!(config.action, GuardrailAction::Block);
    }

    #[test]
    fn test_content_filter_config_builder() {
        let config = ContentFilterConfig::new()
            .threshold(0.9)
            .category(ContentCategory::Sexual)
            .action(GuardrailAction::Warn)
            .strict_mode(true)
            .replacement_text("[REMOVED]");

        assert_eq!(config.threshold, 0.9);
        assert!(config.categories.contains(&ContentCategory::Sexual));
        assert_eq!(config.action, GuardrailAction::Warn);
        assert!(config.strict_mode);
        assert_eq!(config.replacement_text, "[REMOVED]");
    }

    #[test]
    fn test_content_filter_config_threshold_clamp() {
        let config = ContentFilterConfig::new().threshold(1.5);
        assert_eq!(config.threshold, 1.0);

        let config = ContentFilterConfig::new().threshold(-0.5);
        assert_eq!(config.threshold, 0.0);
    }

    #[test]
    fn test_content_filter_config_filters_category() {
        let config = ContentFilterConfig::default();
        assert!(config.filters_category(ContentCategory::Harmful));
        assert!(!config.filters_category(ContentCategory::Sexual));
    }

    #[test]
    fn test_content_filter_config_validate() {
        let valid = ContentFilterConfig::default();
        assert!(valid.validate().is_ok());

        let no_categories = ContentFilterConfig::new().categories(HashSet::new());
        assert!(no_categories.validate().is_err());
    }

    #[test]
    fn test_input_validation_config_default() {
        let config = InputValidationConfig::default();
        assert_eq!(config.max_length, 10000);
        assert_eq!(config.min_length, 1);
        assert!(config.strip_whitespace);
    }

    #[test]
    fn test_input_validation_config_builder() {
        let config = InputValidationConfig::new()
            .max_length(500)
            .min_length(10)
            .allowed_chars("[a-zA-Z0-9]")
            .block_pattern("spam")
            .strip_whitespace(false);

        assert_eq!(config.max_length, 500);
        assert_eq!(config.min_length, 10);
        assert!(config.allowed_chars.is_some());
        assert_eq!(config.blocked_patterns.len(), 1);
        assert!(!config.strip_whitespace);
    }

    #[test]
    fn test_input_validation_validate_input() {
        let config = InputValidationConfig::new()
            .min_length(5)
            .max_length(100)
            .block_pattern("bad");

        assert!(config.validate_input("hello world").is_ok());
        assert!(config.validate_input("hi").is_err()); // too short
        assert!(config.validate_input(&"x".repeat(200)).is_err()); // too long
        assert!(config.validate_input("this is bad").is_err()); // blocked pattern
    }

    #[test]
    fn test_input_validation_strip_whitespace() {
        let config = InputValidationConfig::new()
            .min_length(3)
            .strip_whitespace(true);

        assert!(config.validate_input("   ab   ").is_err()); // "ab" is 2 chars after strip
        assert!(config.validate_input("   abc   ").is_ok()); // "abc" is 3 chars
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_requests, 100);
        assert_eq!(config.window_seconds, 60);
        assert!(!config.allow_burst);
    }

    #[test]
    fn test_rate_limit_config_builder() {
        let config = RateLimitConfig::new()
            .max_requests(50)
            .window_seconds(30)
            .cooldown_seconds(10)
            .allow_burst(true)
            .burst_size(5);

        assert_eq!(config.max_requests, 50);
        assert_eq!(config.window_seconds, 30);
        assert_eq!(config.cooldown_seconds, 10);
        assert!(config.allow_burst);
        assert_eq!(config.burst_size, 5);
    }

    #[test]
    fn test_rate_limit_config_requests_per_second() {
        let config = RateLimitConfig::new()
            .max_requests(120)
            .window_seconds(60);

        assert!((config.requests_per_second() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_rate_limit_config_validate() {
        let valid = RateLimitConfig::default();
        assert!(valid.validate().is_ok());

        let zero_requests = RateLimitConfig::new().max_requests(0);
        assert!(zero_requests.validate().is_err());

        let zero_window = RateLimitConfig::new().window_seconds(0);
        assert!(zero_window.validate().is_err());

        let zero_burst = RateLimitConfig::new().allow_burst(true).burst_size(0);
        assert!(zero_burst.validate().is_err());
    }

    #[test]
    fn test_guardrail_result_pass() {
        let result = GuardrailResult::pass();
        assert!(result.passed);
        assert_eq!(result.action, GuardrailAction::Allow);
    }

    #[test]
    fn test_guardrail_result_block() {
        let result = GuardrailResult::block(ContentCategory::Harmful, 0.95);
        assert!(!result.passed);
        assert_eq!(result.action, GuardrailAction::Block);
        assert!(result.reason.is_some());
    }

    #[test]
    fn test_guardrail_result_warn() {
        let result = GuardrailResult::warn(ContentCategory::Sexual, 0.75);
        assert!(result.passed);
        assert_eq!(result.action, GuardrailAction::Warn);
    }

    #[test]
    fn test_guardrail_result_redact() {
        let result = GuardrailResult::redact(ContentCategory::Illegal, 0.99, "[REDACTED]".to_string());
        assert!(result.passed);
        assert_eq!(result.action, GuardrailAction::Redact);
        assert!(result.modified_content.is_some());
    }

    #[test]
    fn test_guardrail_result_with_category() {
        let result = GuardrailResult::pass()
            .with_category(ContentCategory::Harmful, 0.3)
            .with_category(ContentCategory::Hateful, 0.2);

        assert_eq!(result.categories.len(), 3); // Safe + 2
        assert_eq!(result.scores.len(), 3);
    }

    #[test]
    fn test_guardrail_result_max_score() {
        let result = GuardrailResult::pass()
            .with_category(ContentCategory::Harmful, 0.5)
            .with_category(ContentCategory::Hateful, 0.8);

        assert!((result.max_score() - 1.0).abs() < 0.001); // original Safe has 1.0
    }

    #[test]
    fn test_guardrail_result_most_severe_category() {
        let result = GuardrailResult::pass()
            .with_category(ContentCategory::Sexual, 0.5)
            .with_category(ContentCategory::Illegal, 0.3);

        assert_eq!(result.most_severe_category(), ContentCategory::Illegal);
    }

    #[test]
    fn test_guardrail_stats_new() {
        let stats = GuardrailStats::new();
        assert_eq!(stats.total_checks, 0);
        assert_eq!(stats.pass_rate(), 1.0);
    }

    #[test]
    fn test_guardrail_stats_record() {
        let mut stats = GuardrailStats::new();
        stats.record(&GuardrailResult::pass());
        stats.record(&GuardrailResult::block(ContentCategory::Harmful, 0.9));
        stats.record(&GuardrailResult::warn(ContentCategory::Sexual, 0.7));

        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.passed_checks, 2);
        assert_eq!(stats.blocked_checks, 1);
        assert_eq!(stats.warned_checks, 1);
    }

    #[test]
    fn test_guardrail_stats_rates() {
        let mut stats = GuardrailStats::new();
        stats.record(&GuardrailResult::pass());
        stats.record(&GuardrailResult::pass());
        stats.record(&GuardrailResult::block(ContentCategory::Harmful, 0.9));
        stats.record(&GuardrailResult::block(ContentCategory::Harmful, 0.9));

        assert!((stats.pass_rate() - 0.5).abs() < 0.001);
        assert!((stats.block_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_guardrail_stats_format() {
        let mut stats = GuardrailStats::new();
        stats.record(&GuardrailResult::pass());
        let formatted = stats.format();

        assert!(formatted.contains("1 checks"));
        assert!(formatted.contains("100.0% passed"));
    }

    #[test]
    fn test_calculate_safety_score_empty() {
        assert_eq!(calculate_safety_score(&[], &[]), 1.0);
    }

    #[test]
    fn test_calculate_safety_score_safe() {
        let score = calculate_safety_score(&[ContentCategory::Safe], &[1.0]);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_safety_score_harmful() {
        let score = calculate_safety_score(&[ContentCategory::Illegal], &[1.0]);
        assert!(score < 0.5); // Illegal has severity 10
    }

    #[test]
    fn test_calculate_safety_score_mixed() {
        let score = calculate_safety_score(
            &[ContentCategory::Safe, ContentCategory::Sexual],
            &[1.0, 0.5],
        );
        assert!(score > 0.5);
        assert!(score < 1.0);
    }

    #[test]
    fn test_estimate_moderation_latency() {
        let latency = estimate_moderation_latency(1000, 3);
        assert!(latency > 0);

        let longer_latency = estimate_moderation_latency(5000, 5);
        assert!(longer_latency > latency);
    }
}
