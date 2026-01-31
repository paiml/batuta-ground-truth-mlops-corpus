//! Inference module - Model inference pipelines

mod pipeline;
mod batch;

pub use pipeline::{InferencePipeline, PipelineConfig, PipelineResult};
pub use batch::{BatchProcessor, BatchConfig, BatchResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 64);
    }
}
