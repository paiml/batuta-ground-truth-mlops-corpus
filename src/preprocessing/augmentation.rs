//! Data augmentation utilities

/// Augmentation technique types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugmentationType {
    /// Random word dropout
    WordDropout,
    /// Synonym replacement
    SynonymReplacement,
}

/// Configuration for data augmentation
#[derive(Debug, Clone, Default)]
pub struct AugmentationConfig {
    /// Probability of augmentation
    pub probability: f32,
}

/// Data augmenter
#[derive(Debug, Clone, Default)]
pub struct Augmenter {
    config: AugmentationConfig,
}

impl Augmenter {
    /// Create a new augmenter
    pub fn new(config: AugmentationConfig) -> Self {
        Self { config }
    }

    /// Augment text (placeholder)
    pub fn augment(&self, text: &str) -> String {
        text.to_string()
    }

    /// Get configuration
    pub fn config(&self) -> &AugmentationConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmentation_type_word_dropout() {
        let t = AugmentationType::WordDropout;
        assert_eq!(t, AugmentationType::WordDropout);
    }

    #[test]
    fn test_augmentation_type_synonym_replacement() {
        let t = AugmentationType::SynonymReplacement;
        assert_eq!(t, AugmentationType::SynonymReplacement);
    }

    #[test]
    fn test_augmentation_type_clone() {
        let t = AugmentationType::WordDropout;
        let cloned = t.clone();
        assert_eq!(t, cloned);
    }

    #[test]
    fn test_augmentation_type_debug() {
        let t = AugmentationType::WordDropout;
        let debug = format!("{:?}", t);
        assert!(debug.contains("WordDropout"));
    }

    #[test]
    fn test_augmentation_config_default() {
        let config = AugmentationConfig::default();
        assert_eq!(config.probability, 0.0);
    }

    #[test]
    fn test_augmentation_config_custom() {
        let config = AugmentationConfig { probability: 0.5 };
        assert!((config.probability - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_augmentation_config_clone() {
        let config = AugmentationConfig { probability: 0.3 };
        let cloned = config.clone();
        assert!((cloned.probability - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_augmentation_config_debug() {
        let config = AugmentationConfig { probability: 0.25 };
        let debug = format!("{:?}", config);
        assert!(debug.contains("probability"));
    }

    #[test]
    fn test_augmenter_default() {
        let augmenter = Augmenter::default();
        assert_eq!(augmenter.config().probability, 0.0);
    }

    #[test]
    fn test_augmenter_new() {
        let config = AugmentationConfig { probability: 0.7 };
        let augmenter = Augmenter::new(config);
        assert!((augmenter.config().probability - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_augmenter_augment_preserves_text() {
        let augmenter = Augmenter::default();
        assert_eq!(augmenter.augment("hello"), "hello");
    }

    #[test]
    fn test_augmenter_augment_empty() {
        let augmenter = Augmenter::default();
        assert_eq!(augmenter.augment(""), "");
    }

    #[test]
    fn test_augmenter_augment_long_text() {
        let augmenter = Augmenter::default();
        let text = "This is a longer sentence with multiple words.";
        assert_eq!(augmenter.augment(text), text);
    }

    #[test]
    fn test_augmenter_config_accessor() {
        let config = AugmentationConfig { probability: 0.9 };
        let augmenter = Augmenter::new(config);
        let retrieved = augmenter.config();
        assert!((retrieved.probability - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_augmenter_clone() {
        let config = AugmentationConfig { probability: 0.4 };
        let augmenter = Augmenter::new(config);
        let cloned = augmenter.clone();
        assert!((cloned.config().probability - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_augmenter_debug() {
        let augmenter = Augmenter::default();
        let debug = format!("{:?}", augmenter);
        assert!(debug.contains("Augmenter"));
    }
}
