//! Training demo showing optimizer, scheduler, and training configs

use batuta_ground_truth_mlops_corpus::training::{
    Trainer, TrainerConfig, TrainingMetrics,
    EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, SchedulerType,
};

fn main() {
    println!("=== Training Demo ===\n");

    // Trainer configuration
    println!("Trainer Configuration:");
    let config = TrainerConfig::default()
        .epochs(20)
        .batch_size(32)
        .learning_rate(0.001)
        .weight_decay(0.01)
        .gradient_clip(1.0)
        .early_stopping(5)
        .validation_split(0.2);

    println!("  epochs: {}", config.epochs);
    println!("  batch_size: {}", config.batch_size);
    println!("  learning_rate: {}", config.learning_rate);
    println!("  weight_decay: {}", config.weight_decay);
    println!("  gradient_clip: {:?}", config.gradient_clip);
    println!("  early_stopping_patience: {:?}", config.early_stopping_patience);
    println!("  validation_split: {}", config.validation_split);

    // Create trainer
    println!("\n--- Trainer ---");
    let mut trainer = Trainer::new(config);
    println!("Initial state:");
    println!("  current_epoch: {}", trainer.current_epoch());
    println!("  best_loss: {}", trainer.best_loss());

    // Simulate training epochs
    println!("\n--- Training Progress ---");
    let metrics_history = vec![
        TrainingMetrics { epoch: 1, train_loss: 0.8, val_loss: Some(0.85), train_accuracy: Some(0.65), val_accuracy: Some(0.60) },
        TrainingMetrics { epoch: 2, train_loss: 0.6, val_loss: Some(0.65), train_accuracy: Some(0.75), val_accuracy: Some(0.72) },
        TrainingMetrics { epoch: 3, train_loss: 0.4, val_loss: Some(0.50), train_accuracy: Some(0.85), val_accuracy: Some(0.80) },
        TrainingMetrics { epoch: 4, train_loss: 0.3, val_loss: Some(0.45), train_accuracy: Some(0.90), val_accuracy: Some(0.85) },
        TrainingMetrics { epoch: 5, train_loss: 0.25, val_loss: Some(0.42), train_accuracy: Some(0.92), val_accuracy: Some(0.87) },
    ];

    for metrics in metrics_history {
        println!("Epoch {}: train_loss={:.3}, val_loss={:.3}, train_acc={:.2}, val_acc={:.2}",
            metrics.epoch,
            metrics.train_loss,
            metrics.val_loss.unwrap_or(0.0),
            metrics.train_accuracy.unwrap_or(0.0),
            metrics.val_accuracy.unwrap_or(0.0)
        );
        trainer.record_epoch(metrics);
    }

    println!("\nTraining Summary:");
    println!("  final_epoch: {}", trainer.current_epoch());
    println!("  best_loss: {:.3}", trainer.best_loss());
    println!("  history_length: {}", trainer.history().len());

    // Early stopping
    println!("\n--- Early Stopping ---");
    let mut early_stop = EarlyStopping::new(5, 0.001);
    println!("  patience: {}", early_stop.patience);
    println!("  min_delta: {}", early_stop.min_delta);

    // Simulate early stopping with plateau
    let plateau_metrics = vec![
        TrainingMetrics { epoch: 1, train_loss: 0.5, val_loss: Some(0.5), ..Default::default() },
        TrainingMetrics { epoch: 2, train_loss: 0.5, val_loss: Some(0.5), ..Default::default() },
        TrainingMetrics { epoch: 3, train_loss: 0.5, val_loss: Some(0.5), ..Default::default() },
        TrainingMetrics { epoch: 4, train_loss: 0.5, val_loss: Some(0.5), ..Default::default() },
        TrainingMetrics { epoch: 5, train_loss: 0.5, val_loss: Some(0.5), ..Default::default() },
        TrainingMetrics { epoch: 6, train_loss: 0.5, val_loss: Some(0.5), ..Default::default() },
    ];

    println!("\nSimulating plateau:");
    for m in &plateau_metrics {
        early_stop.on_epoch_end(m);
        println!("  Epoch {}: loss={:.3}, should_stop={}", m.epoch, m.train_loss, early_stop.is_stopped());
        if early_stop.is_stopped() {
            println!("  -> Early stopping triggered!");
            break;
        }
    }

    // Model checkpoint
    println!("\n--- Model Checkpoint ---");
    let checkpoint = ModelCheckpoint::new("val_loss", true);
    println!("  monitor: {}", checkpoint.monitor);
    println!("  save_best_only: {}", checkpoint.save_best_only);

    // Learning rate schedulers
    println!("\n--- Learning Rate Schedulers ---");

    let constant = LearningRateScheduler::constant(0.001);
    println!("Constant Scheduler:");
    println!("  lr: {}", constant.get_lr());
    println!("  type: {:?}", constant.scheduler_type());

    let mut step = LearningRateScheduler::step_decay(0.1, 10, 0.1);
    println!("\nStep Decay Scheduler (lr=0.1, step=10, gamma=0.1):");
    println!("  initial_lr: {}", step.get_lr());
    for i in 0..3 {
        for _ in 0..10 {
            step.step();
        }
        println!("  After {} steps: lr={:.6}", (i + 1) * 10, step.get_lr());
    }

    let mut cosine = LearningRateScheduler::cosine_annealing(0.001, 100, 0.0);
    println!("\nCosine Annealing Scheduler:");
    println!("  initial_lr: {}", cosine.get_lr());
    for checkpoint in [25, 50, 75, 100] {
        while cosine.current_step() < checkpoint {
            cosine.step();
        }
        println!("  Step {}: lr={:.6}", checkpoint, cosine.get_lr());
    }

    // Scheduler types
    println!("\nScheduler Types:");
    println!("  - {:?}", SchedulerType::Constant);
    println!("  - {:?}", SchedulerType::StepDecay);
    println!("  - {:?}", SchedulerType::ExponentialDecay);
    println!("  - {:?}", SchedulerType::CosineAnnealing);
    println!("  - {:?}", SchedulerType::LinearWarmup);
    println!("  - {:?}", SchedulerType::OneCycle);
}
