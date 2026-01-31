# Evaluation API

## Module: `batuta_ground_truth_mlops_corpus::evaluation`

### ClassificationMetrics

```rust
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

impl ClassificationMetrics {
    pub fn new() -> Self;
    pub fn accuracy(self, v: f64) -> Self;
    pub fn precision(self, v: f64) -> Self;
    pub fn recall(self, v: f64) -> Self;
    pub fn f1_score(self, v: f64) -> Self;
}
```

### RegressionMetrics

```rust
pub struct RegressionMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
}

impl RegressionMetrics {
    pub fn new() -> Self;
    pub fn mse(self, v: f64) -> Self;
    pub fn rmse(self, v: f64) -> Self;
    pub fn mae(self, v: f64) -> Self;
    pub fn r2(self, v: f64) -> Self;
}
```

### KFoldConfig

```rust
pub struct KFoldConfig {
    pub n_folds: usize,
    pub shuffle: bool,
    pub random_state: Option<u64>,
}

impl KFoldConfig {
    pub fn new() -> Self;
    pub fn n_folds(self, n: usize) -> Self;
    pub fn shuffle(self, v: bool) -> Self;
    pub fn random_state(self, seed: u64) -> Self;
}
```
