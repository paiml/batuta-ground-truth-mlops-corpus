//! Batch processing for inference

/// Batch configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Enable dynamic batching
    pub dynamic_batching: bool,
    /// Padding strategy
    pub padding: PaddingStrategy,
}

/// Padding strategy for variable-length inputs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to longest in batch
    #[default]
    Longest,
    /// Pad to fixed length
    Fixed(usize),
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            timeout_ms: 1000,
            dynamic_batching: true,
            padding: PaddingStrategy::Longest,
        }
    }
}

impl BatchConfig {
    /// Set max batch size
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set timeout
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Enable/disable dynamic batching
    pub fn dynamic_batching(mut self, enable: bool) -> Self {
        self.dynamic_batching = enable;
        self
    }

    /// Set padding strategy
    pub fn padding(mut self, strategy: PaddingStrategy) -> Self {
        self.padding = strategy;
        self
    }
}

/// Result of batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchResult {
    /// Number of items processed
    pub items_processed: usize,
    /// Number of batches executed
    pub batches_executed: usize,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
    /// Average time per batch
    pub avg_batch_time_ms: f64,
}

impl BatchResult {
    /// Create new batch result
    pub fn new(items: usize, batches: usize, total_ms: u64) -> Self {
        let avg = if batches > 0 {
            total_ms as f64 / batches as f64
        } else {
            0.0
        };
        Self {
            items_processed: items,
            batches_executed: batches,
            total_time_ms: total_ms,
            avg_batch_time_ms: avg,
        }
    }

    /// Calculate throughput (items per second)
    pub fn throughput(&self) -> f64 {
        if self.total_time_ms == 0 {
            0.0
        } else {
            (self.items_processed as f64) / (self.total_time_ms as f64 / 1000.0)
        }
    }
}

/// Batch processor
#[derive(Debug)]
pub struct BatchProcessor {
    config: BatchConfig,
    pending: Vec<Vec<f64>>,
    processed_count: usize,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            processed_count: 0,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Add item to batch queue
    pub fn add(&mut self, item: Vec<f64>) {
        self.pending.push(item);
    }

    /// Add multiple items
    pub fn add_batch(&mut self, items: Vec<Vec<f64>>) {
        self.pending.extend(items);
    }

    /// Get number of pending items
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if batch is ready
    pub fn is_batch_ready(&self) -> bool {
        self.pending.len() >= self.config.max_batch_size
    }

    /// Process pending batch
    pub fn process(&mut self) -> Option<Vec<Vec<f64>>> {
        if self.pending.is_empty() {
            return None;
        }

        let batch_size = self.config.max_batch_size.min(self.pending.len());
        let batch: Vec<Vec<f64>> = self.pending.drain(..batch_size).collect();

        // Apply padding
        let padded = self.apply_padding(batch);
        self.processed_count += padded.len();

        Some(padded)
    }

    /// Force process all pending items
    pub fn flush(&mut self) -> Vec<Vec<Vec<f64>>> {
        let mut batches = Vec::new();

        while !self.pending.is_empty() {
            if let Some(batch) = self.process() {
                batches.push(batch);
            }
        }

        batches
    }

    /// Apply padding strategy
    fn apply_padding(&self, mut batch: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        match self.config.padding {
            PaddingStrategy::None => batch,
            PaddingStrategy::Longest => {
                let max_len = batch.iter().map(|v| v.len()).max().unwrap_or(0);
                for item in &mut batch {
                    item.resize(max_len, 0.0);
                }
                batch
            }
            PaddingStrategy::Fixed(len) => {
                for item in &mut batch {
                    item.resize(len, 0.0);
                }
                batch
            }
        }
    }

    /// Get total processed count
    pub fn processed_count(&self) -> usize {
        self.processed_count
    }

    /// Reset processor
    pub fn reset(&mut self) {
        self.pending.clear();
        self.processed_count = 0;
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(BatchConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.timeout_ms, 1000);
        assert!(config.dynamic_batching);
        assert_eq!(config.padding, PaddingStrategy::Longest);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::default()
            .max_batch_size(128)
            .timeout_ms(500)
            .dynamic_batching(false)
            .padding(PaddingStrategy::Fixed(256));

        assert_eq!(config.max_batch_size, 128);
        assert_eq!(config.timeout_ms, 500);
        assert!(!config.dynamic_batching);
        assert_eq!(config.padding, PaddingStrategy::Fixed(256));
    }

    #[test]
    fn test_batch_result_new() {
        let result = BatchResult::new(100, 10, 500);
        assert_eq!(result.items_processed, 100);
        assert_eq!(result.batches_executed, 10);
        assert_eq!(result.total_time_ms, 500);
        assert!((result.avg_batch_time_ms - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_result_throughput() {
        let result = BatchResult::new(1000, 10, 1000);
        assert!((result.throughput() - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_result_throughput_zero() {
        let result = BatchResult::new(100, 0, 0);
        assert!((result.throughput() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_result_zero_batches() {
        let result = BatchResult::new(0, 0, 0);
        assert!((result.avg_batch_time_ms - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_processor_new() {
        let processor = BatchProcessor::default();
        assert_eq!(processor.pending_count(), 0);
        assert_eq!(processor.processed_count(), 0);
    }

    #[test]
    fn test_batch_processor_add() {
        let mut processor = BatchProcessor::default();
        processor.add(vec![1.0, 2.0, 3.0]);
        assert_eq!(processor.pending_count(), 1);
    }

    #[test]
    fn test_batch_processor_add_batch() {
        let mut processor = BatchProcessor::default();
        processor.add_batch(vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(processor.pending_count(), 3);
    }

    #[test]
    fn test_batch_processor_is_batch_ready() {
        let config = BatchConfig::default().max_batch_size(2);
        let mut processor = BatchProcessor::new(config);

        processor.add(vec![1.0]);
        assert!(!processor.is_batch_ready());

        processor.add(vec![2.0]);
        assert!(processor.is_batch_ready());
    }

    #[test]
    fn test_batch_processor_process() {
        let config = BatchConfig::default().max_batch_size(2);
        let mut processor = BatchProcessor::new(config);

        processor.add_batch(vec![vec![1.0], vec![2.0], vec![3.0]]);

        let batch = processor.process();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
        assert_eq!(processor.pending_count(), 1);
        assert_eq!(processor.processed_count(), 2);
    }

    #[test]
    fn test_batch_processor_process_empty() {
        let mut processor = BatchProcessor::default();
        let batch = processor.process();
        assert!(batch.is_none());
    }

    #[test]
    fn test_batch_processor_flush() {
        let config = BatchConfig::default().max_batch_size(2);
        let mut processor = BatchProcessor::new(config);

        processor.add_batch(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]);

        let batches = processor.flush();
        assert_eq!(batches.len(), 3);
        assert_eq!(processor.pending_count(), 0);
        assert_eq!(processor.processed_count(), 5);
    }

    #[test]
    fn test_batch_processor_padding_longest() {
        let config = BatchConfig::default()
            .max_batch_size(3)
            .padding(PaddingStrategy::Longest);
        let mut processor = BatchProcessor::new(config);

        processor.add_batch(vec![
            vec![1.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0, 3.0],
        ]);

        let batch = processor.process().unwrap();
        assert!(batch.iter().all(|v| v.len() == 3));
    }

    #[test]
    fn test_batch_processor_padding_fixed() {
        let config = BatchConfig::default()
            .max_batch_size(2)
            .padding(PaddingStrategy::Fixed(5));
        let mut processor = BatchProcessor::new(config);

        processor.add_batch(vec![vec![1.0], vec![1.0, 2.0]]);

        let batch = processor.process().unwrap();
        assert!(batch.iter().all(|v| v.len() == 5));
    }

    #[test]
    fn test_batch_processor_padding_none() {
        let config = BatchConfig::default()
            .max_batch_size(2)
            .padding(PaddingStrategy::None);
        let mut processor = BatchProcessor::new(config);

        processor.add_batch(vec![vec![1.0], vec![1.0, 2.0]]);

        let batch = processor.process().unwrap();
        assert_eq!(batch[0].len(), 1);
        assert_eq!(batch[1].len(), 2);
    }

    #[test]
    fn test_batch_processor_reset() {
        let mut processor = BatchProcessor::default();
        processor.add_batch(vec![vec![1.0], vec![2.0]]);

        processor.reset();

        assert_eq!(processor.pending_count(), 0);
        assert_eq!(processor.processed_count(), 0);
    }

    #[test]
    fn test_batch_processor_config_access() {
        let config = BatchConfig::default().max_batch_size(128);
        let processor = BatchProcessor::new(config);
        assert_eq!(processor.config().max_batch_size, 128);
    }

    #[test]
    fn test_padding_strategy_default() {
        let strategy = PaddingStrategy::default();
        assert_eq!(strategy, PaddingStrategy::Longest);
    }

    #[test]
    fn test_padding_strategy_variants() {
        assert_eq!(PaddingStrategy::None, PaddingStrategy::None);
        assert_ne!(PaddingStrategy::None, PaddingStrategy::Longest);
        assert_eq!(PaddingStrategy::Fixed(10), PaddingStrategy::Fixed(10));
        assert_ne!(PaddingStrategy::Fixed(10), PaddingStrategy::Fixed(20));
    }

    #[test]
    fn test_batch_result_default() {
        let result = BatchResult::default();
        assert_eq!(result.items_processed, 0);
        assert_eq!(result.batches_executed, 0);
        assert_eq!(result.total_time_ms, 0);
    }
}
