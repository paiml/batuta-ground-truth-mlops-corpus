# Evaluation Example

Run with: `cargo run --example evaluation_demo`

## Overview

Demonstrates evaluation metrics and cross-validation utilities.

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::evaluation::{
    ClassificationMetrics, RegressionMetrics,
    KFoldConfig, StratifiedKFold,
};

fn main() {
    // Classification metrics
    let clf_metrics = ClassificationMetrics::new()
        .accuracy(0.95)
        .precision(0.94)
        .recall(0.93)
        .f1_score(0.935);

    println!("Accuracy: {:.2}%", clf_metrics.accuracy * 100.0);
    println!("F1 Score: {:.3}", clf_metrics.f1_score);

    // Regression metrics
    let reg_metrics = RegressionMetrics::new()
        .mse(0.0025)
        .rmse(0.05)
        .mae(0.04)
        .r2(0.98);

    println!("R²: {:.2}", reg_metrics.r2);

    // Cross-validation
    let kfold = KFoldConfig::default()
        .n_folds(5)
        .shuffle(true)
        .random_state(42);
}
```

## Metrics Available

### Classification
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

### Regression
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
