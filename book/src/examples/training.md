# Training Example

Run with: `cargo run --example training_demo`

## Overview

Demonstrates training utilities including:
- Trainer configuration
- Training metrics tracking
- Early stopping
- Model checkpointing
- Learning rate schedulers

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::training::{
    TrainerConfig, Trainer, TrainingMetrics,
    EarlyStopping, ModelCheckpoint,
    LearningRateScheduler, SchedulerType,
};

fn main() {
    // Trainer configuration
    let config = TrainerConfig::default()
        .epochs(20)
        .batch_size(32)
        .learning_rate(0.001)
        .weight_decay(0.01)
        .gradient_clip(1.0)
        .early_stopping(5)
        .validation_split(0.2);

    let mut trainer = Trainer::new(config);

    // Record training progress
    let metrics = TrainingMetrics {
        epoch: 1,
        train_loss: 0.8,
        val_loss: Some(0.85),
        train_accuracy: Some(0.65),
        val_accuracy: Some(0.60),
    };
    trainer.record_epoch(metrics);

    // Early stopping
    let mut early_stop = EarlyStopping::new(5, 0.001);
    early_stop.on_epoch_end(&metrics);
    if early_stop.is_stopped() {
        println!("Early stopping triggered!");
    }

    // Learning rate schedulers
    let mut cosine = LearningRateScheduler::cosine_annealing(0.001, 100, 0.0);
    for _ in 0..10 {
        cosine.step();
    }
    println!("Current LR: {}", cosine.get_lr());
}
```

## Scheduler Types

- `Constant` - Fixed learning rate
- `StepDecay` - Reduce by factor every N steps
- `ExponentialDecay` - Exponential decay
- `CosineAnnealing` - Cosine annealing to minimum
- `LinearWarmup` - Linear warmup phase
- `OneCycle` - 1cycle policy
