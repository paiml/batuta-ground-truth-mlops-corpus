//! Deployment module - Model export and quantization

mod export;
mod quantization;

pub use export::{Exporter, ExportConfig, ExportFormat, ExportResult};
pub use quantization::{Quantizer, QuantizationConfig, QuantizationType, QuantizedModel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::Apr);
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quantization_type, QuantizationType::Int8);
    }
}
