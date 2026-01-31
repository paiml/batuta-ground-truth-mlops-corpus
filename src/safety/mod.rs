//! Safety Module
//!
//! Content safety, privacy protection, and model watermarking.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.safety import (
//!     GuardrailType, ContentCategory, ContentFilterConfig,
//!     PiiType, PrivacyConfig, detect_pii,
//!     WatermarkType, WatermarkConfig, embed_watermark,
//! )
//! ```
//!
//! # Submodules
//! - `guardrails`: Content filtering and moderation
//! - `privacy`: PII detection and anonymization
//! - `watermarking`: Model output watermarking

pub mod guardrails;
pub mod privacy;
pub mod watermarking;

// Re-export key types
pub use guardrails::{
    ContentCategory, ContentFilterConfig, GuardrailAction, GuardrailResult, GuardrailStats,
    GuardrailType, InputValidationConfig, RateLimitConfig,
};
pub use privacy::{
    AnonymizationMethod, AnonymizationResult, ComplianceStandard, PiiDetection, PiiType,
    PrivacyConfig, PrivacyStats, RetentionConfig,
};
pub use watermarking::{
    DetectionConfig, DetectionMethod, DetectionResult, EmbedResult, VocabBiasConfig,
    WatermarkConfig, WatermarkStats, WatermarkType,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrails_re_exports() {
        let _ = GuardrailType::default();
        let _ = ContentCategory::default();
        let _ = ContentFilterConfig::default();
    }

    #[test]
    fn test_privacy_re_exports() {
        let _ = PiiType::default();
        let _ = AnonymizationMethod::default();
        let _ = PrivacyConfig::default();
    }

    #[test]
    fn test_watermarking_re_exports() {
        let _ = WatermarkType::default();
        let _ = DetectionMethod::default();
        let _ = WatermarkConfig::default();
    }

    #[test]
    fn test_integration_guardrails_with_privacy() {
        let filter_config = ContentFilterConfig::new()
            .threshold(0.9)
            .action(GuardrailAction::Redact);

        let privacy_config = PrivacyConfig::new()
            .pii_type(PiiType::Email)
            .method(AnonymizationMethod::Mask)
            .standard(ComplianceStandard::Gdpr);

        assert_eq!(filter_config.threshold, 0.9);
        assert!(privacy_config.monitors_type(PiiType::Email));
    }

    #[test]
    fn test_integration_privacy_with_watermarking() {
        let privacy_config = PrivacyConfig::new()
            .threshold(0.8);

        let watermark_config = WatermarkConfig::new()
            .watermark_type(WatermarkType::Soft)
            .strength(0.5);

        assert_eq!(privacy_config.threshold, 0.8);
        assert_eq!(watermark_config.watermark_type, WatermarkType::Soft);
    }

    #[test]
    fn test_comprehensive_safety_pipeline() {
        // Input validation
        let input_config = InputValidationConfig::new()
            .max_length(1000)
            .min_length(1);

        // Content filtering
        let content_config = ContentFilterConfig::new()
            .category(ContentCategory::Harmful)
            .action(GuardrailAction::Block);

        // Privacy protection
        let privacy_config = PrivacyConfig::new()
            .pii_type(PiiType::Email)
            .pii_type(PiiType::Phone);

        // Watermarking
        let watermark_config = WatermarkConfig::new()
            .strength(0.3);

        assert_eq!(input_config.max_length, 1000);
        assert!(content_config.filters_category(ContentCategory::Harmful));
        assert!(privacy_config.monitors_type(PiiType::Email));
        assert_eq!(watermark_config.strength, 0.3);
    }

    #[test]
    fn test_stats_tracking() {
        let mut guardrail_stats = GuardrailStats::new();
        let mut privacy_stats = PrivacyStats::new();
        let mut watermark_stats = WatermarkStats::new();

        // Record various operations
        guardrail_stats.record(&GuardrailResult::pass());
        privacy_stats.record(&AnonymizationResult::new(
            "text".to_string(),
            AnonymizationMethod::Replace,
        ));
        watermark_stats.record_embed(&EmbedResult::success("text".to_string(), 5, 10));

        assert_eq!(guardrail_stats.total_checks, 1);
        assert_eq!(privacy_stats.texts_processed, 1);
        assert_eq!(watermark_stats.texts_processed, 1);
    }
}
