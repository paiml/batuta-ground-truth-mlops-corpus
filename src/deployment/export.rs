//! Model export utilities

use std::path::PathBuf;

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    /// APR v2 format (Sovereign AI Stack native)
    #[default]
    Apr,
    /// JSON format
    Json,
    /// Binary format
    Binary,
    /// ONNX format (metadata only, no runtime)
    OnnxMeta,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Export format
    pub format: ExportFormat,
    /// Output path
    pub output_path: Option<PathBuf>,
    /// Include metadata
    pub include_metadata: bool,
    /// Compression enabled
    pub compress: bool,
    /// Compression level (1-22 for ZSTD)
    pub compression_level: u8,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Apr,
            output_path: None,
            include_metadata: true,
            compress: true,
            compression_level: 3,
        }
    }
}

impl ExportConfig {
    /// Set format
    pub fn format(mut self, fmt: ExportFormat) -> Self {
        self.format = fmt;
        self
    }

    /// Set output path
    pub fn output_path(mut self, path: PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    /// Enable/disable metadata
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Enable/disable compression
    pub fn compress(mut self, enable: bool) -> Self {
        self.compress = enable;
        self
    }

    /// Set compression level
    pub fn compression_level(mut self, level: u8) -> Self {
        self.compression_level = level.min(22);
        self
    }
}

/// Result of export operation
#[derive(Debug, Clone, Default)]
pub struct ExportResult {
    /// Output path
    pub output_path: PathBuf,
    /// Size in bytes
    pub size_bytes: u64,
    /// Compression ratio (if compressed)
    pub compression_ratio: Option<f64>,
    /// Format used
    pub format: ExportFormat,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl ExportResult {
    /// Create successful result
    pub fn success(path: PathBuf, size: u64, format: ExportFormat) -> Self {
        Self {
            output_path: path,
            size_bytes: size,
            compression_ratio: None,
            format,
            success: true,
            error: None,
        }
    }

    /// Create failed result
    pub fn failure(error: &str) -> Self {
        Self {
            success: false,
            error: Some(error.to_string()),
            ..Default::default()
        }
    }

    /// Set compression ratio
    pub fn with_compression_ratio(mut self, ratio: f64) -> Self {
        self.compression_ratio = Some(ratio);
        self
    }

    /// Check if successful
    pub fn is_success(&self) -> bool {
        self.success
    }
}

/// Model exporter
#[derive(Debug)]
pub struct Exporter {
    config: ExportConfig,
}

impl Exporter {
    /// Create new exporter
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &ExportConfig {
        &self.config
    }

    /// Export model weights to bytes
    pub fn export_weights(&self, weights: &[f64]) -> Vec<u8> {
        match self.config.format {
            ExportFormat::Json => self.export_json(weights),
            ExportFormat::Binary => self.export_binary(weights),
            ExportFormat::Apr | ExportFormat::OnnxMeta => self.export_binary(weights),
        }
    }

    fn export_json(&self, weights: &[f64]) -> Vec<u8> {
        let mut output = String::from("[");
        for (i, w) in weights.iter().enumerate() {
            if i > 0 {
                output.push(',');
            }
            output.push_str(&format!("{}", w));
        }
        output.push(']');
        output.into_bytes()
    }

    fn export_binary(&self, weights: &[f64]) -> Vec<u8> {
        let mut output = Vec::with_capacity(weights.len() * 8);
        for w in weights {
            output.extend_from_slice(&w.to_le_bytes());
        }

        if self.config.compress {
            self.compress_data(&output)
        } else {
            output
        }
    }

    fn compress_data(&self, data: &[u8]) -> Vec<u8> {
        // Simple RLE compression for demonstration
        let mut output = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let mut count = 1u8;
            while i + (count as usize) < data.len()
                && count < 255
                && data[i] == data[i + count as usize]
            {
                count += 1;
            }

            output.push(count);
            output.push(data[i]);
            i += count as usize;
        }

        output
    }

    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self.config.format {
            ExportFormat::Apr => ".apr",
            ExportFormat::Json => ".json",
            ExportFormat::Binary => ".bin",
            ExportFormat::OnnxMeta => ".onnx.meta",
        }
    }

    /// Validate export configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.config.compression_level > 22 {
            return Err("Compression level must be <= 22".to_string());
        }
        Ok(())
    }
}

impl Default for Exporter {
    fn default() -> Self {
        Self::new(ExportConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_default() {
        let format = ExportFormat::default();
        assert_eq!(format, ExportFormat::Apr);
    }

    #[test]
    fn test_export_format_variants() {
        assert_eq!(ExportFormat::Apr, ExportFormat::Apr);
        assert_ne!(ExportFormat::Apr, ExportFormat::Json);
        assert_ne!(ExportFormat::Binary, ExportFormat::OnnxMeta);
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::Apr);
        assert!(config.output_path.is_none());
        assert!(config.include_metadata);
        assert!(config.compress);
        assert_eq!(config.compression_level, 3);
    }

    #[test]
    fn test_export_config_builder() {
        let config = ExportConfig::default()
            .format(ExportFormat::Json)
            .output_path(PathBuf::from("/tmp/model.json"))
            .include_metadata(false)
            .compress(false)
            .compression_level(10);

        assert_eq!(config.format, ExportFormat::Json);
        assert_eq!(config.output_path, Some(PathBuf::from("/tmp/model.json")));
        assert!(!config.include_metadata);
        assert!(!config.compress);
        assert_eq!(config.compression_level, 10);
    }

    #[test]
    fn test_export_config_compression_level_clamped() {
        let config = ExportConfig::default().compression_level(100);
        assert_eq!(config.compression_level, 22);
    }

    #[test]
    fn test_export_result_success() {
        let result = ExportResult::success(
            PathBuf::from("/tmp/model.apr"),
            1024,
            ExportFormat::Apr,
        );

        assert!(result.is_success());
        assert_eq!(result.size_bytes, 1024);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_export_result_failure() {
        let result = ExportResult::failure("Export failed");

        assert!(!result.is_success());
        assert_eq!(result.error, Some("Export failed".to_string()));
    }

    #[test]
    fn test_export_result_with_compression() {
        let result = ExportResult::success(
            PathBuf::from("/tmp/model.apr"),
            512,
            ExportFormat::Apr,
        )
        .with_compression_ratio(2.0);

        assert_eq!(result.compression_ratio, Some(2.0));
    }

    #[test]
    fn test_export_result_default() {
        let result = ExportResult::default();
        assert!(!result.is_success());
        assert_eq!(result.size_bytes, 0);
    }

    #[test]
    fn test_exporter_new() {
        let exporter = Exporter::default();
        assert_eq!(exporter.config().format, ExportFormat::Apr);
    }

    #[test]
    fn test_exporter_export_json() {
        let config = ExportConfig::default().format(ExportFormat::Json);
        let exporter = Exporter::new(config);

        let weights = vec![1.0, 2.0, 3.0];
        let output = exporter.export_weights(&weights);
        let json = String::from_utf8(output).unwrap();

        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("1"));
        assert!(json.contains("2"));
        assert!(json.contains("3"));
    }

    #[test]
    fn test_exporter_export_binary() {
        let config = ExportConfig::default()
            .format(ExportFormat::Binary)
            .compress(false);
        let exporter = Exporter::new(config);

        let weights = vec![1.0, 2.0];
        let output = exporter.export_weights(&weights);

        assert_eq!(output.len(), 16); // 2 * 8 bytes
    }

    #[test]
    fn test_exporter_export_binary_compressed() {
        let config = ExportConfig::default()
            .format(ExportFormat::Binary)
            .compress(true);
        let exporter = Exporter::new(config);

        let weights = vec![0.0, 0.0, 0.0, 0.0]; // Compressible data
        let output = exporter.export_weights(&weights);

        // Compressed should be smaller or similar
        assert!(!output.is_empty());
    }

    #[test]
    fn test_exporter_extension() {
        let exporter = Exporter::new(ExportConfig::default().format(ExportFormat::Apr));
        assert_eq!(exporter.extension(), ".apr");

        let exporter = Exporter::new(ExportConfig::default().format(ExportFormat::Json));
        assert_eq!(exporter.extension(), ".json");

        let exporter = Exporter::new(ExportConfig::default().format(ExportFormat::Binary));
        assert_eq!(exporter.extension(), ".bin");

        let exporter = Exporter::new(ExportConfig::default().format(ExportFormat::OnnxMeta));
        assert_eq!(exporter.extension(), ".onnx.meta");
    }

    #[test]
    fn test_exporter_validate() {
        let exporter = Exporter::default();
        assert!(exporter.validate().is_ok());
    }

    #[test]
    fn test_exporter_config_access() {
        let config = ExportConfig::default().format(ExportFormat::Json);
        let exporter = Exporter::new(config);
        assert_eq!(exporter.config().format, ExportFormat::Json);
    }

    #[test]
    fn test_export_empty_weights() {
        let exporter = Exporter::new(ExportConfig::default().format(ExportFormat::Json));
        let output = exporter.export_weights(&[]);
        assert_eq!(output, b"[]");
    }

    #[test]
    fn test_export_apr_format() {
        let exporter = Exporter::new(ExportConfig::default().format(ExportFormat::Apr).compress(false));
        let weights = vec![1.0];
        let output = exporter.export_weights(&weights);
        assert_eq!(output.len(), 8);
    }
}
