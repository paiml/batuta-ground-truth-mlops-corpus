//! Inference pipeline implementation

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Batch size for inference
    pub batch_size: usize,
    /// Number of worker threads
    pub num_workers: usize,
    /// Enable preprocessing
    pub preprocess: bool,
    /// Enable postprocessing
    pub postprocess: bool,
    /// Maximum sequence length
    pub max_length: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 1,
            preprocess: true,
            postprocess: true,
            max_length: 512,
        }
    }
}

impl PipelineConfig {
    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set number of workers
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Enable/disable preprocessing
    pub fn preprocess(mut self, enable: bool) -> Self {
        self.preprocess = enable;
        self
    }

    /// Enable/disable postprocessing
    pub fn postprocess(mut self, enable: bool) -> Self {
        self.postprocess = enable;
        self
    }

    /// Set maximum length
    pub fn max_length(mut self, len: usize) -> Self {
        self.max_length = len;
        self
    }
}

/// Result of inference pipeline
#[derive(Debug, Clone, Default)]
pub struct PipelineResult {
    /// Output scores/logits
    pub scores: Vec<f64>,
    /// Predicted labels (if classification)
    pub labels: Vec<usize>,
    /// Confidence scores
    pub confidences: Vec<f64>,
    /// Processing time in milliseconds
    pub elapsed_ms: u64,
}

impl PipelineResult {
    /// Create new result
    pub fn new(scores: Vec<f64>, labels: Vec<usize>, confidences: Vec<f64>) -> Self {
        Self {
            scores,
            labels,
            confidences,
            elapsed_ms: 0,
        }
    }

    /// Set elapsed time
    pub fn with_elapsed(mut self, ms: u64) -> Self {
        self.elapsed_ms = ms;
        self
    }

    /// Get top prediction
    pub fn top_prediction(&self) -> Option<(usize, f64)> {
        self.labels.first().copied().zip(self.confidences.first().copied())
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Number of predictions
    pub fn len(&self) -> usize {
        self.labels.len()
    }
}

/// Inference pipeline
#[derive(Debug)]
pub struct InferencePipeline {
    config: PipelineConfig,
    is_ready: bool,
}

impl InferencePipeline {
    /// Create new pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            is_ready: false,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Initialize pipeline
    pub fn initialize(&mut self) {
        self.is_ready = true;
    }

    /// Check if ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Run inference on input
    pub fn predict(&self, inputs: &[Vec<f64>]) -> PipelineResult {
        if !self.is_ready {
            return PipelineResult::default();
        }

        let mut scores = Vec::with_capacity(inputs.len());
        let mut labels = Vec::with_capacity(inputs.len());
        let mut confidences = Vec::with_capacity(inputs.len());

        for input in inputs {
            // Simulated inference: max value is prediction
            if let Some((idx, &max_val)) = input
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                labels.push(idx);
                scores.push(max_val);
                // Softmax-style confidence
                let sum: f64 = input.iter().map(|x| x.exp()).sum();
                let confidence = max_val.exp() / sum;
                confidences.push(confidence);
            }
        }

        PipelineResult::new(scores, labels, confidences)
    }

    /// Run inference with preprocessing
    pub fn predict_with_preprocessing(&self, raw_inputs: &[&str]) -> PipelineResult {
        if !self.config.preprocess {
            return PipelineResult::default();
        }

        // Simple preprocessing: convert chars to numeric values
        let inputs: Vec<Vec<f64>> = raw_inputs
            .iter()
            .map(|s| s.chars().map(|c| c as u32 as f64 / 128.0).collect())
            .collect();

        self.predict(&inputs)
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.is_ready = false;
    }
}

impl Default for InferencePipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_workers, 1);
        assert!(config.preprocess);
        assert!(config.postprocess);
        assert_eq!(config.max_length, 512);
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::default()
            .batch_size(64)
            .num_workers(4)
            .preprocess(false)
            .postprocess(false)
            .max_length(1024);

        assert_eq!(config.batch_size, 64);
        assert_eq!(config.num_workers, 4);
        assert!(!config.preprocess);
        assert!(!config.postprocess);
        assert_eq!(config.max_length, 1024);
    }

    #[test]
    fn test_pipeline_result_default() {
        let result = PipelineResult::default();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert!(result.top_prediction().is_none());
    }

    #[test]
    fn test_pipeline_result_new() {
        let result = PipelineResult::new(
            vec![0.9, 0.1],
            vec![0, 1],
            vec![0.9, 0.1],
        );

        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        assert_eq!(result.top_prediction(), Some((0, 0.9)));
    }

    #[test]
    fn test_pipeline_result_with_elapsed() {
        let result = PipelineResult::default().with_elapsed(100);
        assert_eq!(result.elapsed_ms, 100);
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = InferencePipeline::default();
        assert!(!pipeline.is_ready());
    }

    #[test]
    fn test_pipeline_initialize() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();
        assert!(pipeline.is_ready());
    }

    #[test]
    fn test_pipeline_predict_not_ready() {
        let pipeline = InferencePipeline::default();
        let result = pipeline.predict(&[vec![1.0, 2.0, 3.0]]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_predict() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();

        let inputs = vec![vec![1.0, 2.0, 3.0]];
        let result = pipeline.predict(&inputs);

        assert_eq!(result.len(), 1);
        assert_eq!(result.labels[0], 2); // Index of max value
    }

    #[test]
    fn test_pipeline_predict_multiple() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();

        let inputs = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
            vec![1.0, 5.0, 1.0],
        ];
        let result = pipeline.predict(&inputs);

        assert_eq!(result.len(), 3);
        assert_eq!(result.labels[0], 2);
        assert_eq!(result.labels[1], 0);
        assert_eq!(result.labels[2], 1);
    }

    #[test]
    fn test_pipeline_predict_with_preprocessing() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();

        let raw_inputs = vec!["abc", "xyz"];
        let result = pipeline.predict_with_preprocessing(&raw_inputs);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_pipeline_predict_with_preprocessing_disabled() {
        let config = PipelineConfig::default().preprocess(false);
        let mut pipeline = InferencePipeline::new(config);
        pipeline.initialize();

        let raw_inputs = vec!["abc"];
        let result = pipeline.predict_with_preprocessing(&raw_inputs);

        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_reset() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();
        assert!(pipeline.is_ready());

        pipeline.reset();
        assert!(!pipeline.is_ready());
    }

    #[test]
    fn test_pipeline_config_access() {
        let config = PipelineConfig::default().batch_size(128);
        let pipeline = InferencePipeline::new(config);
        assert_eq!(pipeline.config().batch_size, 128);
    }

    #[test]
    fn test_pipeline_empty_input() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();

        let inputs: Vec<Vec<f64>> = vec![];
        let result = pipeline.predict(&inputs);

        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_result_confidences() {
        let mut pipeline = InferencePipeline::default();
        pipeline.initialize();

        let inputs = vec![vec![1.0, 2.0, 3.0]];
        let result = pipeline.predict(&inputs);

        assert!(!result.confidences.is_empty());
        assert!(result.confidences[0] > 0.0);
        assert!(result.confidences[0] <= 1.0);
    }
}
