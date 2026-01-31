//! Tree-based model configurations
//!
//! Provides sklearn-compatible configurations for decision trees and random forests.

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitCriterion {
    /// Gini impurity (default)
    #[default]
    Gini,
    /// Information gain (entropy)
    Entropy,
}

/// Maximum features strategy for random splits
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MaxFeatures {
    /// Use all features
    All,
    /// Use sqrt(n_features) (default)
    #[default]
    Sqrt,
    /// Use log2(n_features)
    Log2,
    /// Use a specific number of features
    Number(usize),
}

/// Configuration for Decision Tree
#[derive(Debug, Clone)]
pub struct DecisionTreeConfig {
    /// Maximum depth of the tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Split criterion (Gini or Entropy)
    pub criterion: SplitCriterion,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            criterion: SplitCriterion::Gini,
            min_samples_split: 2,
        }
    }
}

/// Configuration for Random Forest
///
/// # Python Equivalent (sklearn)
/// ```python
/// from sklearn.ensemble import RandomForestClassifier
/// clf = RandomForestClassifier(n_estimators=100, max_depth=10)
/// ```
#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Maximum depth of each tree
    pub max_depth: Option<usize>,
    /// Split criterion
    pub criterion: SplitCriterion,
    /// Maximum features to consider for splitting
    pub max_features: MaxFeatures,
    /// Number of parallel jobs (-1 for all cores)
    pub n_jobs: i32,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            criterion: SplitCriterion::Gini,
            max_features: MaxFeatures::Sqrt,
            n_jobs: 1,
            random_state: None,
        }
    }
}

impl RandomForestConfig {
    /// Set number of estimators (trees)
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set maximum depth
    pub fn max_depth(mut self, depth: Option<usize>) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set number of parallel jobs
    pub fn n_jobs(mut self, jobs: i32) -> Self {
        self.n_jobs = jobs;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_defaults() {
        let config = RandomForestConfig::default();
        assert_eq!(config.n_estimators, 100);
        assert!(config.max_depth.is_none());
    }

    #[test]
    fn test_builder() {
        let config = RandomForestConfig::default()
            .n_estimators(200)
            .max_depth(Some(15))
            .n_jobs(-1)
            .random_state(42);

        assert_eq!(config.n_estimators, 200);
        assert_eq!(config.max_depth, Some(15));
        assert_eq!(config.n_jobs, -1);
        assert_eq!(config.random_state, Some(42));
    }

    #[test]
    fn test_decision_tree_defaults() {
        let config = DecisionTreeConfig::default();
        assert!(config.max_depth.is_none());
        assert_eq!(config.criterion, SplitCriterion::Gini);
        assert_eq!(config.min_samples_split, 2);
    }

    #[test]
    fn test_split_criterion_default() {
        let criterion = SplitCriterion::default();
        assert_eq!(criterion, SplitCriterion::Gini);
    }

    #[test]
    fn test_max_features_default() {
        let mf = MaxFeatures::default();
        assert_eq!(mf, MaxFeatures::Sqrt);
    }

    #[test]
    fn test_max_features_number() {
        let mf = MaxFeatures::Number(10);
        assert_eq!(mf, MaxFeatures::Number(10));
    }
}
