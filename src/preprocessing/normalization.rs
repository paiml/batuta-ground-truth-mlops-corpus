//! Text normalization utilities

/// Configuration for text normalization
#[derive(Debug, Clone, Default)]
pub struct NormalizerConfig {
    /// Convert to lowercase
    pub lowercase: bool,
    /// Collapse whitespace
    pub collapse_whitespace: bool,
}

/// Text normalizer
#[derive(Debug, Clone, Default)]
pub struct Normalizer {
    config: NormalizerConfig,
}

impl Normalizer {
    /// Create a new normalizer
    pub fn new(config: NormalizerConfig) -> Self {
        Self { config }
    }

    /// Normalize text
    pub fn normalize(&self, text: &str) -> String {
        let mut result = text.trim().to_string();
        if self.config.lowercase {
            result = result.to_lowercase();
        }
        if self.config.collapse_whitespace {
            result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let normalizer = Normalizer::new(NormalizerConfig {
            lowercase: true,
            collapse_whitespace: true,
        });
        assert_eq!(normalizer.normalize("  HELLO  World  "), "hello world");
    }
}
