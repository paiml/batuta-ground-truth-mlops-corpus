//! Training callbacks for monitoring and control

use super::TrainingMetrics;

/// Type of callback event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackType {
    /// Called at the start of training
    OnTrainBegin,
    /// Called at the end of training
    OnTrainEnd,
    /// Called at the start of each epoch
    OnEpochBegin,
    /// Called at the end of each epoch
    OnEpochEnd,
    /// Called at the start of each batch
    OnBatchBegin,
    /// Called at the end of each batch
    OnBatchEnd,
}

/// Callback trait for training events
pub trait Callback: std::fmt::Debug {
    /// Get the callback type
    fn callback_type(&self) -> CallbackType;
    /// Check if training should stop
    fn should_stop(&self) -> bool;
}

/// Early stopping callback
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of epochs without improvement before stopping
    pub patience: usize,
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    best_loss: f64,
    wait: usize,
    stopped: bool,
}

impl EarlyStopping {
    /// Create new early stopping callback
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }

    /// Process epoch end event
    pub fn on_epoch_end(&mut self, metrics: &TrainingMetrics) {
        let current_loss = metrics.val_loss.unwrap_or(metrics.train_loss);

        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
            }
        }
    }

    /// Check if training has stopped
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Reset state for new training run
    pub fn reset(&mut self) {
        self.best_loss = f64::INFINITY;
        self.wait = 0;
        self.stopped = false;
    }

    /// Get current wait count
    pub fn wait_count(&self) -> usize {
        self.wait
    }

    /// Get best loss seen
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

impl Callback for EarlyStopping {
    fn callback_type(&self) -> CallbackType {
        CallbackType::OnEpochEnd
    }

    fn should_stop(&self) -> bool {
        self.stopped
    }
}

/// Model checkpoint callback
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    /// Only save when improvement detected
    pub save_best_only: bool,
    /// Metric to monitor
    pub monitor: String,
    best_value: f64,
    maximize: bool,
    last_saved_epoch: Option<usize>,
}

impl ModelCheckpoint {
    /// Create new model checkpoint callback
    pub fn new(monitor: &str, save_best_only: bool) -> Self {
        let maximize = monitor.contains("accuracy") || monitor.contains("acc");
        Self {
            save_best_only,
            monitor: monitor.to_string(),
            best_value: if maximize { f64::NEG_INFINITY } else { f64::INFINITY },
            maximize,
            last_saved_epoch: None,
        }
    }

    /// Check if model should be saved
    pub fn should_save(&mut self, metrics: &TrainingMetrics) -> bool {
        let current = self.get_monitored_value(metrics);

        if !self.save_best_only {
            self.last_saved_epoch = Some(metrics.epoch);
            return true;
        }

        let improved = if self.maximize {
            current > self.best_value
        } else {
            current < self.best_value
        };

        if improved {
            self.best_value = current;
            self.last_saved_epoch = Some(metrics.epoch);
            true
        } else {
            false
        }
    }

    fn get_monitored_value(&self, metrics: &TrainingMetrics) -> f64 {
        match self.monitor.as_str() {
            "val_loss" => metrics.val_loss.unwrap_or(f64::INFINITY),
            "train_loss" => metrics.train_loss,
            "val_accuracy" | "val_acc" => metrics.val_accuracy.unwrap_or(0.0),
            "train_accuracy" | "train_acc" => metrics.train_accuracy.unwrap_or(0.0),
            _ => metrics.val_loss.unwrap_or(metrics.train_loss),
        }
    }

    /// Get best value seen
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Get last saved epoch
    pub fn last_saved_epoch(&self) -> Option<usize> {
        self.last_saved_epoch
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.best_value = if self.maximize { f64::NEG_INFINITY } else { f64::INFINITY };
        self.last_saved_epoch = None;
    }
}

impl Callback for ModelCheckpoint {
    fn callback_type(&self) -> CallbackType {
        CallbackType::OnEpochEnd
    }

    fn should_stop(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_new() {
        let es = EarlyStopping::new(5, 0.001);
        assert_eq!(es.patience, 5);
        assert!((es.min_delta - 0.001).abs() < 1e-10);
        assert!(!es.is_stopped());
    }

    #[test]
    fn test_early_stopping_improving() {
        let mut es = EarlyStopping::new(3, 0.0);

        for i in 0..5 {
            es.on_epoch_end(&TrainingMetrics {
                epoch: i,
                train_loss: 1.0 - (i as f64 * 0.1),
                val_loss: Some(1.0 - (i as f64 * 0.1)),
                ..Default::default()
            });
        }

        assert!(!es.is_stopped());
    }

    #[test]
    fn test_early_stopping_plateau() {
        let mut es = EarlyStopping::new(3, 0.0);

        for i in 0..5 {
            es.on_epoch_end(&TrainingMetrics {
                epoch: i,
                train_loss: 1.0,
                val_loss: Some(1.0),
                ..Default::default()
            });
        }

        assert!(es.is_stopped());
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut es = EarlyStopping::new(2, 0.0);

        for i in 0..3 {
            es.on_epoch_end(&TrainingMetrics {
                epoch: i,
                train_loss: 1.0,
                ..Default::default()
            });
        }
        assert!(es.is_stopped());

        es.reset();
        assert!(!es.is_stopped());
        assert_eq!(es.wait_count(), 0);
        assert!(es.best_loss().is_infinite());
    }

    #[test]
    fn test_early_stopping_min_delta() {
        let mut es = EarlyStopping::new(2, 0.1);

        es.on_epoch_end(&TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: Some(1.0),
            ..Default::default()
        });

        es.on_epoch_end(&TrainingMetrics {
            epoch: 1,
            train_loss: 0.95,
            val_loss: Some(0.95),
            ..Default::default()
        });

        es.on_epoch_end(&TrainingMetrics {
            epoch: 2,
            train_loss: 0.93,
            val_loss: Some(0.93),
            ..Default::default()
        });

        assert!(es.is_stopped());
    }

    #[test]
    fn test_early_stopping_uses_train_loss() {
        let mut es = EarlyStopping::new(2, 0.0);

        es.on_epoch_end(&TrainingMetrics {
            epoch: 0,
            train_loss: 0.5,
            val_loss: None,
            ..Default::default()
        });

        assert!((es.best_loss() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_callback_type() {
        let es = EarlyStopping::new(5, 0.001);
        assert_eq!(es.callback_type(), CallbackType::OnEpochEnd);
    }

    #[test]
    fn test_model_checkpoint_new() {
        let cp = ModelCheckpoint::new("val_loss", true);
        assert!(cp.save_best_only);
        assert_eq!(cp.monitor, "val_loss");
    }

    #[test]
    fn test_model_checkpoint_save_best_loss() {
        let mut cp = ModelCheckpoint::new("val_loss", true);

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: Some(0.8),
            ..Default::default()
        }));

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 1,
            train_loss: 0.9,
            val_loss: Some(0.6),
            ..Default::default()
        }));

        assert!(!cp.should_save(&TrainingMetrics {
            epoch: 2,
            train_loss: 0.85,
            val_loss: Some(0.7),
            ..Default::default()
        }));
    }

    #[test]
    fn test_model_checkpoint_save_best_accuracy() {
        let mut cp = ModelCheckpoint::new("val_accuracy", true);

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_accuracy: Some(0.7),
            ..Default::default()
        }));

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 1,
            train_loss: 0.9,
            val_accuracy: Some(0.8),
            ..Default::default()
        }));

        assert!(!cp.should_save(&TrainingMetrics {
            epoch: 2,
            train_loss: 0.85,
            val_accuracy: Some(0.75),
            ..Default::default()
        }));
    }

    #[test]
    fn test_model_checkpoint_always_save() {
        let mut cp = ModelCheckpoint::new("val_loss", false);

        for i in 0..3 {
            assert!(cp.should_save(&TrainingMetrics {
                epoch: i,
                train_loss: 1.0,
                val_loss: Some(1.0),
                ..Default::default()
            }));
        }
    }

    #[test]
    fn test_model_checkpoint_reset() {
        let mut cp = ModelCheckpoint::new("val_loss", true);

        cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 0.5,
            val_loss: Some(0.5),
            ..Default::default()
        });

        cp.reset();

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: Some(1.0),
            ..Default::default()
        }));
        assert!(cp.last_saved_epoch().is_none() || cp.last_saved_epoch() == Some(0));
    }

    #[test]
    fn test_model_checkpoint_callback_trait() {
        let cp = ModelCheckpoint::new("val_loss", true);
        assert_eq!(cp.callback_type(), CallbackType::OnEpochEnd);
        assert!(!cp.should_stop());
    }

    #[test]
    fn test_model_checkpoint_train_accuracy() {
        let mut cp = ModelCheckpoint::new("train_acc", true);

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            train_accuracy: Some(0.7),
            ..Default::default()
        }));

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 1,
            train_loss: 0.9,
            train_accuracy: Some(0.8),
            ..Default::default()
        }));
    }

    #[test]
    fn test_model_checkpoint_train_loss() {
        let mut cp = ModelCheckpoint::new("train_loss", true);

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 0.8,
            ..Default::default()
        }));

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 1,
            train_loss: 0.6,
            ..Default::default()
        }));

        assert!(!cp.should_save(&TrainingMetrics {
            epoch: 2,
            train_loss: 0.7,
            ..Default::default()
        }));
    }

    #[test]
    fn test_model_checkpoint_unknown_metric() {
        let mut cp = ModelCheckpoint::new("unknown", true);

        assert!(cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 0.5,
            val_loss: Some(0.5),
            ..Default::default()
        }));
    }

    #[test]
    fn test_callback_type_variants() {
        assert_eq!(CallbackType::OnTrainBegin, CallbackType::OnTrainBegin);
        assert_ne!(CallbackType::OnTrainBegin, CallbackType::OnTrainEnd);
        assert_ne!(CallbackType::OnEpochBegin, CallbackType::OnEpochEnd);
        assert_ne!(CallbackType::OnBatchBegin, CallbackType::OnBatchEnd);
    }

    #[test]
    fn test_model_checkpoint_best_value() {
        let mut cp = ModelCheckpoint::new("val_loss", true);

        cp.should_save(&TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: Some(0.5),
            ..Default::default()
        });

        assert!((cp.best_value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_model_checkpoint_last_saved() {
        let mut cp = ModelCheckpoint::new("val_loss", true);
        assert!(cp.last_saved_epoch().is_none());

        cp.should_save(&TrainingMetrics {
            epoch: 5,
            train_loss: 1.0,
            val_loss: Some(0.5),
            ..Default::default()
        });

        assert_eq!(cp.last_saved_epoch(), Some(5));
    }

    #[test]
    fn test_early_stopping_wait_count() {
        let mut es = EarlyStopping::new(5, 0.0);

        es.on_epoch_end(&TrainingMetrics {
            epoch: 0,
            train_loss: 0.5,
            ..Default::default()
        });

        es.on_epoch_end(&TrainingMetrics {
            epoch: 1,
            train_loss: 0.5,
            ..Default::default()
        });

        assert_eq!(es.wait_count(), 1);
    }
}
