//! Model architectures and configurations

mod tree;
mod boosting;
mod transformer;

pub use tree::{RandomForestConfig, DecisionTreeConfig, SplitCriterion, MaxFeatures};
pub use boosting::GradientBoostingConfig;
pub use transformer::TransformerConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_forest_config() {
        let config = RandomForestConfig::default()
            .n_estimators(100)
            .max_depth(Some(10));
        assert_eq!(config.n_estimators, 100);
        assert_eq!(config.max_depth, Some(10));
    }
}
