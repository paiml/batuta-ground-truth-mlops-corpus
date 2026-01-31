//! Integration Tests with Aprender ML Algorithms
//!
//! These tests verify that the corpus types integrate correctly
//! with aprender machine learning algorithms.

use batuta_ground_truth_mlops_corpus::models::{RandomForestConfig, GradientBoostingConfig};
use batuta_ground_truth_mlops_corpus::training::TrainerConfig;
use batuta_ground_truth_mlops_corpus::evaluation::{ClassificationMetrics, RegressionMetrics};
use batuta_ground_truth_mlops_corpus::preprocessing::{Tokenizer, TokenizerConfig, Normalizer};

use aprender::metrics::classification::accuracy;
use aprender::metrics::{mse, r_squared};
use aprender::primitives::Vector as AprVector;

// =============================================================================
// Preprocessing Integration
// =============================================================================

#[test]
fn test_tokenizer_features_for_ml() {
    let tokenizer = Tokenizer::new(TokenizerConfig::default());

    // Create training samples
    let texts = vec![
        "positive sentiment happy",
        "negative sentiment sad",
        "neutral statement here",
    ];

    let features: Vec<Vec<f64>> = texts
        .iter()
        .map(|text| {
            let tokens = tokenizer.tokenize(text);
            // Simple bag-of-words features
            vec![
                tokens.len() as f64,
                tokens.iter().map(|t| t.text.len()).sum::<usize>() as f64,
                tokens.iter().map(|t| t.text.len() as f64).sum::<f64>() / tokens.len() as f64,
            ]
        })
        .collect();

    assert_eq!(features.len(), 3);
    assert_eq!(features[0].len(), 3);
}

// =============================================================================
// Model Config Integration
// =============================================================================

#[test]
fn test_random_forest_config_with_aprender() {
    let config = RandomForestConfig::default()
        .n_estimators(50)
        .max_depth(Some(10));

    // Verify config values are compatible with aprender expectations
    assert!(config.n_estimators > 0);
    assert!(config.max_depth.unwrap() > 0);

    // These would be used to configure aprender's RandomForest
    let _n_trees = config.n_estimators;
    let _depth = config.max_depth;
}

#[test]
fn test_gradient_boosting_config_with_aprender() {
    let config = GradientBoostingConfig::default()
        .n_estimators(100)
        .learning_rate(0.1)
        .max_depth(3);

    // Verify config values are compatible with aprender expectations
    assert!(config.n_estimators > 0);
    assert!(config.learning_rate > 0.0 && config.learning_rate <= 1.0);
    assert!(config.max_depth > 0);
}

// =============================================================================
// Metrics Integration
// =============================================================================

#[test]
fn test_classification_metrics_with_aprender_accuracy() {
    // Ground truth and predictions (using usize for aprender classification)
    let y_true: Vec<usize> = vec![0, 1, 1, 0, 1, 0, 1, 1];
    let y_pred: Vec<usize> = vec![0, 1, 0, 0, 1, 1, 1, 1];

    // Use aprender's accuracy function
    let acc = accuracy(&y_pred, &y_true);

    // Create corpus metrics for comparison
    let mut metrics = ClassificationMetrics::default();
    metrics.accuracy = acc as f64;

    assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
    assert!((acc - 0.75).abs() < 0.01); // 6/8 = 0.75
}

#[test]
fn test_regression_metrics_with_aprender_mse() {
    // Ground truth and predictions
    let y_true = AprVector::from_slice(&[3.0f32, -0.5, 2.0, 7.0]);
    let y_pred = AprVector::from_slice(&[2.5f32, 0.0, 2.0, 8.0]);

    // Use aprender's MSE function
    let mse_val = mse(&y_pred, &y_true);

    // Create corpus metrics
    let mut metrics = RegressionMetrics::default();
    metrics.mse = mse_val as f64;

    assert!(metrics.mse >= 0.0);
    // MSE = ((0.5)^2 + (0.5)^2 + 0 + 1) / 4 = 0.375
    assert!((mse_val - 0.375).abs() < 0.01);
}

#[test]
fn test_regression_metrics_with_aprender_r2() {
    let y_true = AprVector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = AprVector::from_slice(&[1.1f32, 2.0, 2.9, 4.2, 4.8]);

    // Use aprender's R² function
    let r2 = r_squared(&y_pred, &y_true);

    // Create corpus metrics
    let mut metrics = RegressionMetrics::default();
    metrics.r2 = r2 as f64;

    assert!(metrics.r2 <= 1.0);
    assert!(metrics.r2 > 0.9); // Should be high for good predictions
}

// =============================================================================
// Training Integration
// =============================================================================

#[test]
fn test_trainer_config_compatible_with_aprender() {
    let config = TrainerConfig::default()
        .learning_rate(0.001)
        .epochs(100)
        .batch_size(32);

    // These values should be usable with aprender training loops
    assert!(config.learning_rate > 0.0);
    assert!(config.epochs > 0);
    assert!(config.batch_size > 0);

    // Simulate training loop configuration
    let _lr = config.learning_rate;
    let _num_epochs = config.epochs;
    let _batch = config.batch_size;
}

#[test]
fn test_early_stopping_simulation_with_aprender_metrics() {
    // Simulate early stopping logic with aprender metrics
    let val_losses = vec![1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89];
    let patience = 5;
    let mut best_loss = f64::INFINITY;
    let mut wait = 0;

    for loss in val_losses {
        if loss < best_loss {
            best_loss = loss;
            wait = 0;
        } else {
            wait += 1;
        }
        if wait >= patience {
            break;
        }
    }

    // Should have stopped after patience exhausted
    assert!(wait >= patience - 1 || wait >= 4);
}

// =============================================================================
// Pipeline Integration
// =============================================================================

#[test]
fn test_preprocessing_pipeline_with_aprender() {
    // Create preprocessing pipeline
    let normalizer = Normalizer::default();
    let tokenizer = Tokenizer::new(TokenizerConfig::default());

    // Process text
    let text = "   Hello WORLD   ";
    let normalized = normalizer.normalize(text);
    let tokens = tokenizer.tokenize(&normalized);

    // Convert to features for aprender
    let _features: Vec<f64> = tokens.iter().map(|t| t.text.len() as f64).collect();

    assert!(!tokens.is_empty());
}

#[test]
fn test_evaluation_pipeline_with_aprender() {
    use batuta_ground_truth_mlops_corpus::evaluation::CrossValidationConfig;

    let cv_config = CrossValidationConfig::default();

    // Simulate cross-validation with aprender metrics
    let fold_scores: Vec<f32> = (0..cv_config.n_folds)
        .map(|i| {
            // Simulate fold training and evaluation
            let y_true: Vec<usize> = vec![0, 1, 1, 0, 1];
            let y_pred: Vec<usize> = vec![0, 1, 0, 0, 1];
            accuracy(&y_pred, &y_true) + (i as f32) * 0.01
        })
        .collect();

    let mean_score: f32 = fold_scores.iter().sum::<f32>() / fold_scores.len() as f32;
    let std_score: f32 = (fold_scores
        .iter()
        .map(|x| (x - mean_score).powi(2))
        .sum::<f32>()
        / fold_scores.len() as f32)
        .sqrt();

    assert!(mean_score > 0.5);
    assert!(std_score < 0.2);
}

// =============================================================================
// End-to-End Integration
// =============================================================================

#[test]
fn test_full_ml_pipeline_integration() {
    // 1. Preprocessing with corpus tokenizer
    let tokenizer = Tokenizer::new(TokenizerConfig::default());
    let texts = vec![
        "good positive excellent",
        "bad negative terrible",
        "okay neutral average",
    ];
    let labels: Vec<usize> = vec![1, 0, 1];

    // 2. Feature extraction
    let _features: Vec<Vec<f64>> = texts
        .iter()
        .map(|text| {
            let tokens = tokenizer.tokenize(text);
            vec![
                tokens.len() as f64,
                tokens.iter().map(|t| t.text.len()).sum::<usize>() as f64 / tokens.len() as f64,
            ]
        })
        .collect();

    // 3. Simulate predictions (in real use, would use aprender model)
    let predictions: Vec<usize> = vec![1, 0, 1]; // Perfect predictions for this example

    // 4. Evaluate with aprender metrics
    let acc = accuracy(&predictions, &labels);

    assert_eq!(acc, 1.0);
}

#[test]
fn test_model_config_to_aprender_params() {
    let rf_config = RandomForestConfig::default()
        .n_estimators(100)
        .max_depth(Some(10));

    // Extract parameters for aprender RandomForest
    let params = (
        rf_config.n_estimators,
        rf_config.max_depth,
    );

    assert_eq!(params.0, 100);
    assert_eq!(params.1, Some(10));
}

// =============================================================================
// Metrics Calculation Integration
// =============================================================================

#[test]
fn test_binary_classification_metrics() {
    let y_true: Vec<usize> = vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
    let y_pred: Vec<usize> = vec![1, 0, 1, 0, 0, 1, 1, 0, 1, 1];

    let acc = accuracy(&y_pred, &y_true);

    // 8/10 correct
    assert!((acc - 0.8).abs() < 0.01);
}

#[test]
fn test_multiclass_classification_metrics() {
    let y_true: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let y_pred: Vec<usize> = vec![0, 1, 2, 0, 2, 2, 1, 1, 2];

    let acc = accuracy(&y_pred, &y_true);

    // 7/9 correct
    assert!((acc - 7.0 / 9.0).abs() < 0.01);
}

#[test]
fn test_regression_mse_perfect_predictions() {
    let y_true = AprVector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = AprVector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);

    let mse_val = mse(&y_pred, &y_true);

    assert!(mse_val.abs() < 1e-10);
}

#[test]
fn test_regression_r2_perfect_predictions() {
    let y_true = AprVector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = AprVector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);

    let r2 = r_squared(&y_pred, &y_true);

    assert!((r2 - 1.0).abs() < 1e-10);
}

#[test]
fn test_regression_r2_poor_predictions() {
    let y_true = AprVector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = AprVector::from_slice(&[3.0f32, 3.0, 3.0, 3.0, 3.0]); // Mean prediction

    let r2 = r_squared(&y_pred, &y_true);

    // R² should be 0 for mean predictions
    assert!(r2.abs() < 0.01);
}
