//! Training module - Model training utilities and configurations

mod trainer;
mod callbacks;
mod schedulers;

pub use trainer::{Trainer, TrainerConfig, TrainingMetrics};
pub use callbacks::{Callback, CallbackType, EarlyStopping, ModelCheckpoint};
pub use schedulers::{LearningRateScheduler, SchedulerType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_defaults() {
        let config = TrainerConfig::default();
        assert_eq!(config.epochs, 10);
        assert!(config.learning_rate > 0.0);
    }

    #[test]
    fn test_early_stopping_new() {
        let es = EarlyStopping::new(5, 0.001);
        assert_eq!(es.patience, 5);
    }

    #[test]
    fn test_scheduler_constant() {
        let scheduler = LearningRateScheduler::constant(0.01);
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);
    }
}
