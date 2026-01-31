//! Evaluation and metrics demo

use batuta_ground_truth_mlops_corpus::evaluation::{
    ClassificationMetrics, RegressionMetrics, CrossValidationConfig,
};

fn main() {
    println!("=== Evaluation & Metrics Demo ===\n");

    // Classification metrics
    println!("Classification Metrics:");
    let mut clf_metrics = ClassificationMetrics::default();
    clf_metrics.accuracy = 0.95;
    clf_metrics.precision = 0.94;
    clf_metrics.recall = 0.93;
    clf_metrics.f1 = 0.935;

    println!("  accuracy:  {:.2}%", clf_metrics.accuracy * 100.0);
    println!("  precision: {:.2}%", clf_metrics.precision * 100.0);
    println!("  recall:    {:.2}%", clf_metrics.recall * 100.0);
    println!("  f1:        {:.3}", clf_metrics.f1);

    // Regression metrics
    println!("\n--- Regression Metrics ---");
    let mut reg_metrics = RegressionMetrics::default();
    reg_metrics.mse = 0.0025;
    reg_metrics.rmse = 0.05;
    reg_metrics.mae = 0.04;
    reg_metrics.r2 = 0.98;

    println!("  MSE:  {:.4}", reg_metrics.mse);
    println!("  RMSE: {:.4}", reg_metrics.rmse);
    println!("  MAE:  {:.4}", reg_metrics.mae);
    println!("  R²:   {:.2}", reg_metrics.r2);

    // Cross-validation configuration
    println!("\n--- Cross-Validation ---");
    let cv_config = CrossValidationConfig::default();
    println!("Default K-Fold Configuration:");
    println!("  n_folds: {}", cv_config.n_folds);
    println!("  shuffle: {}", cv_config.shuffle);
    println!("  random_state: {:?}", cv_config.random_state);

    // Simulate cross-validation scores
    println!("\n--- Simulated CV Results ---");
    let fold_scores = [0.94, 0.96, 0.93, 0.95, 0.97];
    let mean: f64 = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let variance: f64 = fold_scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / fold_scores.len() as f64;
    let std = variance.sqrt();

    println!("Fold scores: {:?}", fold_scores);
    println!("Mean accuracy: {:.2}% (+/- {:.2}%)", mean * 100.0, std * 200.0);
}
