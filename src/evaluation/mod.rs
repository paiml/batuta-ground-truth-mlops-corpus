//! Evaluation module - Metrics and cross-validation

mod metrics;
mod cross_validation;

pub use metrics::{
    Metrics, ClassificationMetrics, RegressionMetrics,
    accuracy, precision, recall, f1_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix,
};
pub use cross_validation::{CrossValidator, CrossValidationConfig, FoldResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let acc = accuracy(&y_true, &y_pred);
        assert!((acc - 2.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let mse = mean_squared_error(&y_true, &y_pred);
        assert!((mse - 0.375).abs() < 1e-10);
    }
}
