//! Learning rate schedulers

/// Type of learning rate scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerType {
    /// Constant learning rate
    #[default]
    Constant,
    /// Step decay
    StepDecay,
    /// Exponential decay
    ExponentialDecay,
    /// Cosine annealing
    CosineAnnealing,
    /// Linear warmup
    LinearWarmup,
    /// One cycle policy
    OneCycle,
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    scheduler_type: SchedulerType,
    initial_lr: f64,
    current_lr: f64,
    step_size: usize,
    gamma: f64,
    min_lr: f64,
    max_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LearningRateScheduler {
    /// Create constant scheduler
    pub fn constant(lr: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::Constant,
            initial_lr: lr,
            current_lr: lr,
            step_size: 1,
            gamma: 1.0,
            min_lr: 0.0,
            max_lr: lr,
            warmup_steps: 0,
            total_steps: 0,
            current_step: 0,
        }
    }

    /// Create step decay scheduler
    pub fn step_decay(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::StepDecay,
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            min_lr: 0.0,
            max_lr: initial_lr,
            warmup_steps: 0,
            total_steps: 0,
            current_step: 0,
        }
    }

    /// Create exponential decay scheduler
    pub fn exponential_decay(initial_lr: f64, gamma: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::ExponentialDecay,
            initial_lr,
            current_lr: initial_lr,
            step_size: 1,
            gamma,
            min_lr: 0.0,
            max_lr: initial_lr,
            warmup_steps: 0,
            total_steps: 0,
            current_step: 0,
        }
    }

    /// Create cosine annealing scheduler
    pub fn cosine_annealing(initial_lr: f64, total_steps: usize, min_lr: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::CosineAnnealing,
            initial_lr,
            current_lr: initial_lr,
            step_size: 1,
            gamma: 1.0,
            min_lr,
            max_lr: initial_lr,
            warmup_steps: 0,
            total_steps,
            current_step: 0,
        }
    }

    /// Create linear warmup scheduler
    pub fn linear_warmup(target_lr: f64, warmup_steps: usize) -> Self {
        Self {
            scheduler_type: SchedulerType::LinearWarmup,
            initial_lr: 0.0,
            current_lr: 0.0,
            step_size: 1,
            gamma: 1.0,
            min_lr: 0.0,
            max_lr: target_lr,
            warmup_steps,
            total_steps: warmup_steps,
            current_step: 0,
        }
    }

    /// Create one cycle scheduler
    pub fn one_cycle(max_lr: f64, total_steps: usize) -> Self {
        Self {
            scheduler_type: SchedulerType::OneCycle,
            initial_lr: max_lr / 25.0,
            current_lr: max_lr / 25.0,
            step_size: 1,
            gamma: 1.0,
            min_lr: max_lr / 1000.0,
            max_lr,
            warmup_steps: total_steps / 3,
            total_steps,
            current_step: 0,
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Get scheduler type
    pub fn scheduler_type(&self) -> SchedulerType {
        self.scheduler_type
    }

    /// Get current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get total steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Step the scheduler
    pub fn step(&mut self) {
        self.current_step += 1;
        self.current_lr = self.calculate_lr();
    }

    fn calculate_lr(&self) -> f64 {
        match self.scheduler_type {
            SchedulerType::Constant => self.initial_lr,
            SchedulerType::StepDecay => {
                let num_decays = self.current_step / self.step_size;
                self.initial_lr * self.gamma.powi(num_decays as i32)
            }
            SchedulerType::ExponentialDecay => {
                self.initial_lr * self.gamma.powi(self.current_step as i32)
            }
            SchedulerType::CosineAnnealing => {
                if self.total_steps == 0 {
                    return self.initial_lr;
                }
                let progress = (self.current_step as f64) / (self.total_steps as f64);
                let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                self.min_lr + (self.initial_lr - self.min_lr) * cosine
            }
            SchedulerType::LinearWarmup => {
                if self.warmup_steps == 0 {
                    return self.max_lr;
                }
                let progress = (self.current_step as f64) / (self.warmup_steps as f64);
                self.max_lr * progress.min(1.0)
            }
            SchedulerType::OneCycle => {
                if self.total_steps == 0 {
                    return self.max_lr;
                }
                let progress = (self.current_step as f64) / (self.total_steps as f64);

                if progress < 0.3 {
                    let warmup_progress = progress / 0.3;
                    self.initial_lr + (self.max_lr - self.initial_lr) * warmup_progress
                } else {
                    let anneal_progress = (progress - 0.3) / 0.7;
                    let cosine = (1.0 + (std::f64::consts::PI * anneal_progress).cos()) / 2.0;
                    self.min_lr + (self.max_lr - self.min_lr) * cosine
                }
            }
        }
    }

    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = match self.scheduler_type {
            SchedulerType::LinearWarmup => 0.0,
            SchedulerType::OneCycle => self.initial_lr,
            _ => self.initial_lr,
        };
    }
}

impl Default for LearningRateScheduler {
    fn default() -> Self {
        Self::constant(0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let mut scheduler = LearningRateScheduler::constant(0.01);
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);

        scheduler.step();
        scheduler.step();
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_step_decay_scheduler() {
        let mut scheduler = LearningRateScheduler::step_decay(0.1, 2, 0.5);
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);

        scheduler.step();
        scheduler.step();
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-10);

        scheduler.step();
        scheduler.step();
        assert!((scheduler.get_lr() - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay_scheduler() {
        let mut scheduler = LearningRateScheduler::exponential_decay(1.0, 0.9);
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-10);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.9).abs() < 1e-10);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let mut scheduler = LearningRateScheduler::cosine_annealing(1.0, 100, 0.0);
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-10);

        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.5).abs() < 0.01);

        for _ in 0..50 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.01);
    }

    #[test]
    fn test_linear_warmup_scheduler() {
        let mut scheduler = LearningRateScheduler::linear_warmup(1.0, 10);
        assert!(scheduler.get_lr() < 0.01);

        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.5).abs() < 0.01);

        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_one_cycle_scheduler() {
        let mut scheduler = LearningRateScheduler::one_cycle(1.0, 100);
        let initial = scheduler.get_lr();
        assert!(initial < 1.0);

        for _ in 0..30 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 1.0).abs() < 0.05);

        for _ in 0..70 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.01);
    }

    #[test]
    fn test_scheduler_reset() {
        let mut scheduler = LearningRateScheduler::step_decay(0.1, 2, 0.5);

        scheduler.step();
        scheduler.step();
        scheduler.step();

        scheduler.reset();
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
        assert_eq!(scheduler.current_step(), 0);
    }

    #[test]
    fn test_scheduler_type() {
        let scheduler = LearningRateScheduler::constant(0.01);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::Constant);

        let scheduler = LearningRateScheduler::step_decay(0.1, 2, 0.5);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::StepDecay);
    }

    #[test]
    fn test_scheduler_type_default() {
        let st = SchedulerType::default();
        assert_eq!(st, SchedulerType::Constant);
    }

    #[test]
    fn test_scheduler_default() {
        let scheduler = LearningRateScheduler::default();
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-10);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::Constant);
    }

    #[test]
    fn test_cosine_annealing_zero_steps() {
        let scheduler = LearningRateScheduler::cosine_annealing(1.0, 0, 0.0);
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_warmup_zero_steps() {
        let mut scheduler = LearningRateScheduler::linear_warmup(1.0, 0);
        scheduler.step(); // Need to step to trigger calculate_lr
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_one_cycle_zero_steps() {
        let mut scheduler = LearningRateScheduler::one_cycle(1.0, 0);
        scheduler.step(); // Need to step to trigger calculate_lr
        assert!((scheduler.get_lr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scheduler_type_variants() {
        assert_eq!(SchedulerType::Constant, SchedulerType::Constant);
        assert_ne!(SchedulerType::Constant, SchedulerType::StepDecay);
        assert_ne!(SchedulerType::ExponentialDecay, SchedulerType::CosineAnnealing);
        assert_ne!(SchedulerType::LinearWarmup, SchedulerType::OneCycle);
    }

    #[test]
    fn test_scheduler_total_steps() {
        let scheduler = LearningRateScheduler::cosine_annealing(1.0, 100, 0.0);
        assert_eq!(scheduler.total_steps(), 100);
    }

    #[test]
    fn test_linear_warmup_reset() {
        let mut scheduler = LearningRateScheduler::linear_warmup(1.0, 10);

        for _ in 0..5 {
            scheduler.step();
        }

        scheduler.reset();
        assert!(scheduler.get_lr() < 0.01);
    }

    #[test]
    fn test_one_cycle_reset() {
        let mut scheduler = LearningRateScheduler::one_cycle(1.0, 100);
        let initial = scheduler.get_lr();

        for _ in 0..50 {
            scheduler.step();
        }

        scheduler.reset();
        assert!((scheduler.get_lr() - initial).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay_type() {
        let scheduler = LearningRateScheduler::exponential_decay(1.0, 0.9);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::ExponentialDecay);
    }

    #[test]
    fn test_cosine_type() {
        let scheduler = LearningRateScheduler::cosine_annealing(1.0, 100, 0.0);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::CosineAnnealing);
    }

    #[test]
    fn test_linear_warmup_type() {
        let scheduler = LearningRateScheduler::linear_warmup(1.0, 10);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::LinearWarmup);
    }

    #[test]
    fn test_one_cycle_type() {
        let scheduler = LearningRateScheduler::one_cycle(1.0, 100);
        assert_eq!(scheduler.scheduler_type(), SchedulerType::OneCycle);
    }
}
