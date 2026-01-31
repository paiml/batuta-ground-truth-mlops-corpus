//! Evaluation metrics for ML models

use std::collections::HashMap;

/// Container for various metrics
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    values: HashMap<String, f64>,
}

impl Metrics {
    /// Create new metrics container
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metric
    pub fn add(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }

    /// Get a metric by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }

    /// Get all metric names
    pub fn names(&self) -> Vec<&str> {
        self.values.keys().map(|s| s.as_str()).collect()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Number of metrics
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Classification metrics
#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    /// Accuracy score
    pub accuracy: f64,
    /// Precision (macro average)
    pub precision: f64,
    /// Recall (macro average)
    pub recall: f64,
    /// F1 score (macro average)
    pub f1: f64,
    /// Per-class metrics
    pub per_class: HashMap<usize, ClassMetrics>,
}

/// Per-class metrics
#[derive(Debug, Clone, Default)]
pub struct ClassMetrics {
    /// Precision for this class
    pub precision: f64,
    /// Recall for this class
    pub recall: f64,
    /// F1 for this class
    pub f1: f64,
    /// Support (number of samples)
    pub support: usize,
}

impl ClassificationMetrics {
    /// Compute classification metrics from predictions
    pub fn compute(y_true: &[usize], y_pred: &[usize]) -> Self {
        let acc = accuracy(y_true, y_pred);

        // Get unique classes
        let mut classes: Vec<usize> = y_true.iter().chain(y_pred.iter()).copied().collect();
        classes.sort();
        classes.dedup();

        let mut per_class = HashMap::new();
        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut total_f1 = 0.0;

        for &class in &classes {
            let (tp, fp, fn_count) = compute_tp_fp_fn(y_true, y_pred, class);
            let support = y_true.iter().filter(|&&y| y == class).count();

            let prec = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let rec = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let f1_val = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };

            per_class.insert(class, ClassMetrics {
                precision: prec,
                recall: rec,
                f1: f1_val,
                support,
            });

            total_precision += prec;
            total_recall += rec;
            total_f1 += f1_val;
        }

        let n_classes = classes.len() as f64;
        Self {
            accuracy: acc,
            precision: if n_classes > 0.0 { total_precision / n_classes } else { 0.0 },
            recall: if n_classes > 0.0 { total_recall / n_classes } else { 0.0 },
            f1: if n_classes > 0.0 { total_f1 / n_classes } else { 0.0 },
            per_class,
        }
    }
}

fn compute_tp_fp_fn(y_true: &[usize], y_pred: &[usize], class: usize) -> (usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        if *pred_val == class {
            if *true_val == class {
                tp += 1;
            } else {
                fp += 1;
            }
        } else if *true_val == class {
            fn_count += 1;
        }
    }

    (tp, fp, fn_count)
}

/// Regression metrics
#[derive(Debug, Clone, Default)]
pub struct RegressionMetrics {
    /// Mean squared error
    pub mse: f64,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute error
    pub mae: f64,
    /// R-squared score
    pub r2: f64,
}

impl RegressionMetrics {
    /// Compute regression metrics from predictions
    pub fn compute(y_true: &[f64], y_pred: &[f64]) -> Self {
        let mse = mean_squared_error(y_true, y_pred);
        let mae = mean_absolute_error(y_true, y_pred);
        let r2 = r2_score(y_true, y_pred);

        Self {
            mse,
            rmse: mse.sqrt(),
            mae,
            r2,
        }
    }
}

/// Calculate accuracy score
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Calculate precision for binary classification
pub fn precision(y_true: &[usize], y_pred: &[usize], positive_class: usize) -> f64 {
    let (tp, fp, _) = compute_tp_fp_fn(y_true, y_pred, positive_class);
    if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    }
}

/// Calculate recall for binary classification
pub fn recall(y_true: &[usize], y_pred: &[usize], positive_class: usize) -> f64 {
    let (tp, _, fn_count) = compute_tp_fp_fn(y_true, y_pred, positive_class);
    if tp + fn_count == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_count) as f64
    }
}

/// Calculate F1 score for binary classification
pub fn f1_score(y_true: &[usize], y_pred: &[usize], positive_class: usize) -> f64 {
    let prec = precision(y_true, y_pred, positive_class);
    let rec = recall(y_true, y_pred, positive_class);

    if prec + rec == 0.0 {
        0.0
    } else {
        2.0 * prec * rec / (prec + rec)
    }
}

/// Calculate mean squared error
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    sum / y_true.len() as f64
}

/// Calculate mean absolute error
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum();
    sum / y_true.len() as f64
}

/// Calculate R-squared score
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }

    let mean: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;

    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();

    if ss_tot == 0.0 {
        return 1.0; // Perfect prediction if all true values are the same
    }

    1.0 - ss_res / ss_tot
}

/// Calculate confusion matrix
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize]) -> Vec<Vec<usize>> {
    if y_true.is_empty() {
        return Vec::new();
    }

    let max_class = y_true.iter().chain(y_pred.iter()).max().copied().unwrap_or(0);
    let n_classes = max_class + 1;

    let mut matrix = vec![vec![0usize; n_classes]; n_classes];

    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        matrix[*true_val][*pred_val] += 1;
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 1, 2, 0, 1, 2];
        assert!((accuracy(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_zero() {
        let y_true = vec![0, 0, 0];
        let y_pred = vec![1, 1, 1];
        assert!((accuracy(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_empty() {
        let y_true: Vec<usize> = vec![];
        let y_pred: Vec<usize> = vec![];
        assert!((accuracy(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision() {
        let y_true = vec![1, 1, 0, 0, 1, 0];
        let y_pred = vec![1, 0, 0, 0, 1, 1];
        let prec = precision(&y_true, &y_pred, 1);
        assert!((prec - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_no_positives() {
        let y_true = vec![0, 0, 0];
        let y_pred = vec![0, 0, 0];
        assert!((precision(&y_true, &y_pred, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall() {
        let y_true = vec![1, 1, 0, 0, 1, 0];
        let y_pred = vec![1, 0, 0, 0, 1, 1];
        let rec = recall(&y_true, &y_pred, 1);
        assert!((rec - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_no_true_positives() {
        let y_true = vec![0, 0, 0];
        let y_pred = vec![1, 1, 1];
        assert!((recall(&y_true, &y_pred, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score() {
        let y_true = vec![1, 1, 0, 0, 1, 0];
        let y_pred = vec![1, 1, 0, 0, 0, 0];
        let f1 = f1_score(&y_true, &y_pred, 1);
        // precision = 2/2 = 1.0, recall = 2/3 = 0.667
        // f1 = 2 * 1.0 * 0.667 / 1.667 = 0.8
        assert!((f1 - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score_zero() {
        let y_true = vec![0, 0, 0];
        let y_pred = vec![0, 0, 0];
        assert!((f1_score(&y_true, &y_pred, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let mse = mean_squared_error(&y_true, &y_pred);
        // (0.5^2 + 0.5^2 + 0 + 1) / 4 = 0.375
        assert!((mse - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_mse_empty() {
        let y_true: Vec<f64> = vec![];
        let y_pred: Vec<f64> = vec![];
        assert!((mean_squared_error(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let mae = mean_absolute_error(&y_true, &y_pred);
        // (0.5 + 0.5 + 0 + 1) / 4 = 0.5
        assert!((mae - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mae_empty() {
        let y_true: Vec<f64> = vec![];
        let y_pred: Vec<f64> = vec![];
        assert!((mean_absolute_error(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        assert!((r2_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_bad() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![3.0, 2.0, 1.0];
        let r2 = r2_score(&y_true, &y_pred);
        assert!(r2 < 0.0); // Worse than mean predictor
    }

    #[test]
    fn test_r2_empty() {
        let y_true: Vec<f64> = vec![];
        let y_pred: Vec<f64> = vec![];
        assert!((r2_score(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_constant() {
        let y_true = vec![2.0, 2.0, 2.0];
        let y_pred = vec![2.0, 2.0, 2.0];
        assert!((r2_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 2];

        let cm = confusion_matrix(&y_true, &y_pred);

        assert_eq!(cm[0][0], 2); // True 0, Pred 0
        assert_eq!(cm[1][0], 1); // True 1, Pred 0
        assert_eq!(cm[1][2], 1); // True 1, Pred 2
        assert_eq!(cm[2][1], 1); // True 2, Pred 1
        assert_eq!(cm[2][2], 1); // True 2, Pred 2
    }

    #[test]
    fn test_confusion_matrix_empty() {
        let y_true: Vec<usize> = vec![];
        let y_pred: Vec<usize> = vec![];
        let cm = confusion_matrix(&y_true, &y_pred);
        assert!(cm.is_empty());
    }

    #[test]
    fn test_classification_metrics_compute() {
        let y_true = vec![0, 0, 1, 1, 2, 2];
        let y_pred = vec![0, 0, 1, 2, 2, 2];

        let metrics = ClassificationMetrics::compute(&y_true, &y_pred);

        assert!((metrics.accuracy - 5.0 / 6.0).abs() < 1e-10);
        assert!(metrics.precision > 0.0);
        assert!(metrics.recall > 0.0);
        assert!(metrics.f1 > 0.0);
    }

    #[test]
    fn test_classification_metrics_per_class() {
        let y_true = vec![0, 0, 1, 1];
        let y_pred = vec![0, 0, 1, 0];

        let metrics = ClassificationMetrics::compute(&y_true, &y_pred);

        assert!(metrics.per_class.contains_key(&0));
        assert!(metrics.per_class.contains_key(&1));

        let class_0 = &metrics.per_class[&0];
        assert_eq!(class_0.support, 2);
    }

    #[test]
    fn test_regression_metrics_compute() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];

        let metrics = RegressionMetrics::compute(&y_true, &y_pred);

        assert!((metrics.mse - 0.375).abs() < 1e-10);
        assert!((metrics.rmse - 0.375_f64.sqrt()).abs() < 1e-10);
        assert!((metrics.mae - 0.5).abs() < 1e-10);
        assert!(metrics.r2 > 0.0);
    }

    #[test]
    fn test_metrics_container() {
        let mut metrics = Metrics::new();
        assert!(metrics.is_empty());

        metrics.add("accuracy", 0.95);
        metrics.add("f1", 0.92);

        assert_eq!(metrics.len(), 2);
        assert!(!metrics.is_empty());
        assert_eq!(metrics.get("accuracy"), Some(0.95));
        assert_eq!(metrics.get("f1"), Some(0.92));
        assert!(metrics.get("precision").is_none());
    }

    #[test]
    fn test_metrics_names() {
        let mut metrics = Metrics::new();
        metrics.add("a", 1.0);
        metrics.add("b", 2.0);

        let names = metrics.names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_class_metrics_default() {
        let cm = ClassMetrics::default();
        assert!((cm.precision - 0.0).abs() < 1e-10);
        assert!((cm.recall - 0.0).abs() < 1e-10);
        assert!((cm.f1 - 0.0).abs() < 1e-10);
        assert_eq!(cm.support, 0);
    }

    #[test]
    fn test_regression_metrics_default() {
        let rm = RegressionMetrics::default();
        assert!((rm.mse - 0.0).abs() < 1e-10);
        assert!((rm.rmse - 0.0).abs() < 1e-10);
        assert!((rm.mae - 0.0).abs() < 1e-10);
        assert!((rm.r2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification_metrics_default() {
        let cm = ClassificationMetrics::default();
        assert!((cm.accuracy - 0.0).abs() < 1e-10);
        assert!(cm.per_class.is_empty());
    }

    #[test]
    fn test_classification_metrics_empty() {
        let y_true: Vec<usize> = vec![];
        let y_pred: Vec<usize> = vec![];
        let metrics = ClassificationMetrics::compute(&y_true, &y_pred);
        assert!((metrics.accuracy - 0.0).abs() < 1e-10);
    }
}
