# Models API

## Module: `batuta_ground_truth_mlops_corpus::models`

### RandomForestConfig

```rust
pub struct RandomForestConfig {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub n_jobs: i32,
    pub random_state: Option<u64>,
    pub criterion: SplitCriterion,
    pub max_features: MaxFeatures,
}

impl RandomForestConfig {
    pub fn new() -> Self;
    pub fn n_estimators(self, n: usize) -> Self;
    pub fn max_depth(self, d: usize) -> Self;
    pub fn n_jobs(self, n: i32) -> Self;
    pub fn random_state(self, seed: u64) -> Self;
}
```

### TransformerConfig

```rust
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl TransformerConfig {
    pub fn bert_base() -> Self;
    pub fn bert_large() -> Self;
    pub fn gpt2() -> Self;
    pub fn gpt2_medium() -> Self;
    pub fn gpt2_large() -> Self;
}
```

### GradientBoostingConfig

```rust
pub struct GradientBoostingConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
}
```
