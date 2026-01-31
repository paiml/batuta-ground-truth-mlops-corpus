//! Core trainer implementation

/// Training metrics tracked during training
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Current epoch number
    pub epoch: usize,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss (if validation data provided)
    pub val_loss: Option<f64>,
    /// Training accuracy (if classification)
    pub train_accuracy: Option<f64>,
    /// Validation accuracy (if classification)
    pub val_accuracy: Option<f64>,
}

/// Trainer configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
    /// Validation split ratio
    pub validation_split: f64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.001,
            weight_decay: 0.0,
            gradient_clip: None,
            early_stopping_patience: None,
            validation_split: 0.2,
        }
    }
}

impl TrainerConfig {
    /// Set number of epochs
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set gradient clipping
    pub fn gradient_clip(mut self, clip: f64) -> Self {
        self.gradient_clip = Some(clip);
        self
    }

    /// Set early stopping patience
    pub fn early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = Some(patience);
        self
    }

    /// Set validation split
    pub fn validation_split(mut self, split: f64) -> Self {
        self.validation_split = split;
        self
    }
}

/// Generic trainer for ML models
#[derive(Debug)]
pub struct Trainer {
    config: TrainerConfig,
    current_epoch: usize,
    best_loss: f64,
    history: Vec<TrainingMetrics>,
}

impl Trainer {
    /// Create a new trainer with config
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            current_epoch: 0,
            best_loss: f64::INFINITY,
            history: Vec::new(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get best loss seen so far
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }

    /// Get training history
    pub fn history(&self) -> &[TrainingMetrics] {
        &self.history
    }

    /// Record metrics for an epoch
    pub fn record_epoch(&mut self, metrics: TrainingMetrics) {
        let loss = metrics.val_loss.unwrap_or(metrics.train_loss);
        if loss < self.best_loss {
            self.best_loss = loss;
        }
        self.current_epoch = metrics.epoch;
        self.history.push(metrics);
    }

    /// Check if training should stop early
    pub fn should_stop(&self) -> bool {
        if let Some(patience) = self.config.early_stopping_patience {
            if self.history.len() < patience {
                return false;
            }
            let recent: Vec<_> = self.history.iter().rev().take(patience).collect();
            let losses: Vec<f64> = recent
                .iter()
                .map(|m| m.val_loss.unwrap_or(m.train_loss))
                .collect();

            if let Some(&first) = losses.last() {
                losses.iter().all(|&l| l >= first)
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Reset trainer state for new training run
    pub fn reset(&mut self) {
        self.current_epoch = 0;
        self.best_loss = f64::INFINITY;
        self.history.clear();
    }
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new(TrainerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_builder() {
        let config = TrainerConfig::default()
            .epochs(20)
            .batch_size(64)
            .learning_rate(0.01)
            .weight_decay(0.001)
            .gradient_clip(1.0)
            .early_stopping(5)
            .validation_split(0.1);

        assert_eq!(config.epochs, 20);
        assert_eq!(config.batch_size, 64);
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
        assert!((config.weight_decay - 0.001).abs() < 1e-10);
        assert_eq!(config.gradient_clip, Some(1.0));
        assert_eq!(config.early_stopping_patience, Some(5));
        assert!((config.validation_split - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_new() {
        let trainer = Trainer::default();
        assert_eq!(trainer.current_epoch(), 0);
        assert!(trainer.best_loss().is_infinite());
        assert!(trainer.history().is_empty());
    }

    #[test]
    fn test_trainer_record_epoch() {
        let mut trainer = Trainer::default();

        trainer.record_epoch(TrainingMetrics {
            epoch: 1,
            train_loss: 0.5,
            val_loss: Some(0.6),
            train_accuracy: Some(0.8),
            val_accuracy: Some(0.75),
        });

        assert_eq!(trainer.current_epoch(), 1);
        assert!((trainer.best_loss() - 0.6).abs() < 1e-10);
        assert_eq!(trainer.history().len(), 1);
    }

    #[test]
    fn test_trainer_best_loss_updates() {
        let mut trainer = Trainer::default();

        trainer.record_epoch(TrainingMetrics {
            epoch: 1,
            train_loss: 0.5,
            val_loss: Some(0.6),
            ..Default::default()
        });

        trainer.record_epoch(TrainingMetrics {
            epoch: 2,
            train_loss: 0.4,
            val_loss: Some(0.4),
            ..Default::default()
        });

        assert!((trainer.best_loss() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_uses_train_loss_when_no_val() {
        let mut trainer = Trainer::default();

        trainer.record_epoch(TrainingMetrics {
            epoch: 1,
            train_loss: 0.3,
            val_loss: None,
            ..Default::default()
        });

        assert!((trainer.best_loss() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_should_stop_no_patience() {
        let trainer = Trainer::default();
        assert!(!trainer.should_stop());
    }

    #[test]
    fn test_trainer_should_stop_not_enough_epochs() {
        let config = TrainerConfig::default().early_stopping(5);
        let mut trainer = Trainer::new(config);

        for i in 0..3 {
            trainer.record_epoch(TrainingMetrics {
                epoch: i,
                train_loss: 1.0,
                ..Default::default()
            });
        }

        assert!(!trainer.should_stop());
    }

    #[test]
    fn test_trainer_should_stop_improving() {
        let config = TrainerConfig::default().early_stopping(3);
        let mut trainer = Trainer::new(config);

        for i in 0..5 {
            trainer.record_epoch(TrainingMetrics {
                epoch: i,
                train_loss: 1.0 - (i as f64 * 0.1),
                ..Default::default()
            });
        }

        assert!(!trainer.should_stop());
    }

    #[test]
    fn test_trainer_should_stop_plateau() {
        let config = TrainerConfig::default().early_stopping(3);
        let mut trainer = Trainer::new(config);

        for i in 0..5 {
            trainer.record_epoch(TrainingMetrics {
                epoch: i,
                train_loss: 1.0,
                ..Default::default()
            });
        }

        assert!(trainer.should_stop());
    }

    #[test]
    fn test_trainer_reset() {
        let mut trainer = Trainer::default();

        trainer.record_epoch(TrainingMetrics {
            epoch: 1,
            train_loss: 0.5,
            ..Default::default()
        });

        trainer.reset();

        assert_eq!(trainer.current_epoch(), 0);
        assert!(trainer.best_loss().is_infinite());
        assert!(trainer.history().is_empty());
    }

    #[test]
    fn test_training_metrics_default() {
        let metrics = TrainingMetrics::default();
        assert_eq!(metrics.epoch, 0);
        assert!((metrics.train_loss - 0.0).abs() < 1e-10);
        assert!(metrics.val_loss.is_none());
        assert!(metrics.train_accuracy.is_none());
        assert!(metrics.val_accuracy.is_none());
    }

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
        assert!((config.learning_rate - 0.001).abs() < 1e-10);
        assert!((config.weight_decay - 0.0).abs() < 1e-10);
        assert!(config.gradient_clip.is_none());
        assert!(config.early_stopping_patience.is_none());
        assert!((config.validation_split - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_trainer_empty_history_stop() {
        let config = TrainerConfig::default().early_stopping(3);
        let trainer = Trainer::new(config);
        assert!(!trainer.should_stop());
    }
}
