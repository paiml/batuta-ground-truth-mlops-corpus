# Model Configuration Example

Run with: `cargo run --example random_forest_demo`

## Overview

Demonstrates configuration patterns for ML models including Random Forest, Gradient Boosting, and Transformers.

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::models::{
    RandomForestConfig, DecisionTreeConfig,
    GradientBoostingConfig, TransformerConfig,
};

fn main() {
    // Random Forest Configuration
    let rf = RandomForestConfig::default()
        .n_estimators(200)
        .max_depth(15)
        .n_jobs(-1)
        .random_state(42);

    println!("n_estimators: {}", rf.n_estimators);
    println!("max_depth: {:?}", rf.max_depth);

    // Decision Tree
    let dt = DecisionTreeConfig::default();
    println!("criterion: {:?}", dt.criterion);

    // Gradient Boosting
    let gb = GradientBoostingConfig::default()
        .learning_rate(0.1)
        .n_estimators(100);

    // Transformer Configurations
    let bert_base = TransformerConfig::bert_base();
    let bert_large = TransformerConfig::bert_large();
    let gpt2 = TransformerConfig::gpt2();
}
```

## Output

```
=== Model Configuration Demo ===

Random Forest Configuration:
  n_estimators: 200
  max_depth: Some(15)
  n_jobs: -1
  random_state: Some(42)
  criterion: Gini
  max_features: Sqrt

Transformer Configurations:
BERT Base:
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 12

BERT Large:
  hidden_size: 1024
  num_attention_heads: 16
  num_hidden_layers: 24
```
