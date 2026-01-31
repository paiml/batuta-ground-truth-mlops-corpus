//! Random Forest configuration example demonstrating model configurations

use batuta_ground_truth_mlops_corpus::models::{
    RandomForestConfig, DecisionTreeConfig, SplitCriterion, MaxFeatures,
    GradientBoostingConfig, TransformerConfig,
};

fn main() {
    println!("=== Model Configuration Demo ===\n");

    // Random Forest configuration (sklearn-compatible)
    println!("Random Forest Configuration:");
    let rf_config = RandomForestConfig::default()
        .n_estimators(200)
        .max_depth(Some(15))
        .n_jobs(-1)
        .random_state(42);

    println!("  n_estimators: {}", rf_config.n_estimators);
    println!("  max_depth: {:?}", rf_config.max_depth);
    println!("  n_jobs: {}", rf_config.n_jobs);
    println!("  random_state: {:?}", rf_config.random_state);
    println!("  criterion: {:?}", rf_config.criterion);
    println!("  max_features: {:?}", rf_config.max_features);

    // Decision Tree configuration
    println!("\nDecision Tree Configuration:");
    let dt_config = DecisionTreeConfig::default();
    println!("  max_depth: {:?}", dt_config.max_depth);
    println!("  criterion: {:?}", dt_config.criterion);
    println!("  min_samples_split: {}", dt_config.min_samples_split);

    // Gradient Boosting configuration
    println!("\nGradient Boosting Configuration:");
    let gb_config = GradientBoostingConfig::default();
    println!("  n_estimators: {}", gb_config.n_estimators);
    println!("  learning_rate: {}", gb_config.learning_rate);
    println!("  max_depth: {}", gb_config.max_depth);

    // Transformer configuration
    println!("\nTransformer Configurations:");

    let bert_base = TransformerConfig::bert_base();
    println!("BERT Base:");
    println!("  hidden_size: {}", bert_base.hidden_size);
    println!("  num_attention_heads: {}", bert_base.num_attention_heads);
    println!("  num_hidden_layers: {}", bert_base.num_hidden_layers);
    println!("  vocab_size: {}", bert_base.vocab_size);

    let bert_large = TransformerConfig::bert_large();
    println!("\nBERT Large:");
    println!("  hidden_size: {}", bert_large.hidden_size);
    println!("  num_attention_heads: {}", bert_large.num_attention_heads);
    println!("  num_hidden_layers: {}", bert_large.num_hidden_layers);
    println!("  vocab_size: {}", bert_large.vocab_size);
}
