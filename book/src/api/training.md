# Training API

## Module: `batuta_ground_truth_mlops_corpus::training`

### TrainerConfig

```rust
pub struct TrainerConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub gradient_clip: Option<f64>,
    pub early_stopping_patience: Option<usize>,
    pub validation_split: f64,
}

impl TrainerConfig {
    pub fn new() -> Self;
    pub fn epochs(self, n: usize) -> Self;
    pub fn batch_size(self, n: usize) -> Self;
    pub fn learning_rate(self, lr: f64) -> Self;
    pub fn weight_decay(self, wd: f64) -> Self;
    pub fn gradient_clip(self, gc: f64) -> Self;
    pub fn early_stopping(self, patience: usize) -> Self;
    pub fn validation_split(self, split: f64) -> Self;
}
```

### Trainer

```rust
pub struct Trainer { ... }

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self;
    pub fn record_epoch(&mut self, metrics: TrainingMetrics);
    pub fn current_epoch(&self) -> usize;
    pub fn best_loss(&self) -> f64;
    pub fn history(&self) -> &[TrainingMetrics];
}
```

### TrainingMetrics

```rust
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub train_accuracy: Option<f64>,
    pub val_accuracy: Option<f64>,
}
```

### LearningRateScheduler

```rust
pub struct LearningRateScheduler { ... }

impl LearningRateScheduler {
    pub fn constant(lr: f64) -> Self;
    pub fn step_decay(initial_lr: f64, step_size: usize, gamma: f64) -> Self;
    pub fn cosine_annealing(initial_lr: f64, t_max: usize, eta_min: f64) -> Self;
    pub fn step(&mut self);
    pub fn get_lr(&self) -> f64;
    pub fn current_step(&self) -> usize;
    pub fn scheduler_type(&self) -> SchedulerType;
}
```

### EarlyStopping

```rust
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self;
    pub fn on_epoch_end(&mut self, metrics: &TrainingMetrics);
    pub fn is_stopped(&self) -> bool;
}
```
