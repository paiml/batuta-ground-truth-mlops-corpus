//! Gradient Boosting model configurations
//!
//! Provides sklearn-compatible configurations for gradient boosting.

/// Configuration for Gradient Boosting
///
/// # Python Equivalent (sklearn)
/// ```python
/// from sklearn.ensemble import GradientBoostingClassifier
/// clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
/// ```
#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    /// Number of boosting stages
    pub n_estimators: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Maximum depth of individual trees
    pub max_depth: usize,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
        }
    }
}

impl GradientBoostingConfig {
    /// Set number of estimators
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let config = GradientBoostingConfig::default();
        assert_eq!(config.n_estimators, 100);
        assert!((config.learning_rate - 0.1).abs() < 1e-10);
        assert_eq!(config.max_depth, 3);
    }

    #[test]
    fn test_builder() {
        let config = GradientBoostingConfig::default()
            .n_estimators(200)
            .learning_rate(0.05)
            .max_depth(5);

        assert_eq!(config.n_estimators, 200);
        assert!((config.learning_rate - 0.05).abs() < 1e-10);
        assert_eq!(config.max_depth, 5);
    }
}
