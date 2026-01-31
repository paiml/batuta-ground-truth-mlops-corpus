//! Dataset Management
//!
//! Provides dataset configuration, format handling, and statistics.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.hub import DatasetFormat, create_dataset_config, SplitType
//! config = create_dataset_config(name="my-dataset", format=DatasetFormat.PARQUET)
//! ```

/// Dataset format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DatasetFormat {
    /// Parquet columnar format (default)
    #[default]
    Parquet,
    /// CSV format
    Csv,
    /// JSON Lines format
    JsonLines,
    /// Arrow IPC format
    Arrow,
    /// Text files
    Text,
}

impl DatasetFormat {
    /// Get format name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::Csv => "csv",
            Self::JsonLines => "jsonl",
            Self::Arrow => "arrow",
            Self::Text => "text",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Parquet => ".parquet",
            Self::Csv => ".csv",
            Self::JsonLines => ".jsonl",
            Self::Arrow => ".arrow",
            Self::Text => ".txt",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "parquet" => Some(Self::Parquet),
            "csv" => Some(Self::Csv),
            "jsonl" | "jsonlines" | "json_lines" => Some(Self::JsonLines),
            "arrow" | "ipc" => Some(Self::Arrow),
            "text" | "txt" => Some(Self::Text),
            _ => None,
        }
    }

    /// List all formats
    pub fn list_all() -> Vec<Self> {
        vec![Self::Parquet, Self::Csv, Self::JsonLines, Self::Arrow, Self::Text]
    }

    /// Check if format supports compression
    pub fn supports_compression(&self) -> bool {
        matches!(self, Self::Parquet | Self::Arrow)
    }
}

/// Dataset split types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitType {
    /// Training split (default)
    #[default]
    Train,
    /// Validation split
    Validation,
    /// Test split
    Test,
    /// All data
    All,
}

impl SplitType {
    /// Get split name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
            Self::All => "all",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "train" | "training" => Some(Self::Train),
            "validation" | "valid" | "val" | "dev" => Some(Self::Validation),
            "test" | "testing" => Some(Self::Test),
            "all" | "*" => Some(Self::All),
            _ => None,
        }
    }

    /// List all splits
    pub fn list_all() -> Vec<Self> {
        vec![Self::Train, Self::Validation, Self::Test, Self::All]
    }
}

/// Streaming mode for large datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamingMode {
    /// Load entire dataset into memory
    #[default]
    Disabled,
    /// Stream from source
    Enabled,
    /// Auto-detect based on size
    Auto,
}

impl StreamingMode {
    /// Get mode name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::Auto => "auto",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "disabled" | "off" | "false" => Some(Self::Disabled),
            "enabled" | "on" | "true" => Some(Self::Enabled),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    /// List all modes
    pub fn list_all() -> Vec<Self> {
        vec![Self::Disabled, Self::Enabled, Self::Auto]
    }
}

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Dataset name
    pub name: String,
    /// Dataset format
    pub format: DatasetFormat,
    /// Default split
    pub split: SplitType,
    /// Streaming mode
    pub streaming: StreamingMode,
    /// Number of workers for loading
    pub num_workers: usize,
    /// Shuffle data
    pub shuffle: bool,
    /// Random seed
    pub seed: Option<u64>,
    /// Batch size
    pub batch_size: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            format: DatasetFormat::Parquet,
            split: SplitType::Train,
            streaming: StreamingMode::Disabled,
            num_workers: 4,
            shuffle: true,
            seed: None,
            batch_size: 32,
        }
    }
}

impl DatasetConfig {
    /// Create new dataset config
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set format
    pub fn format(mut self, fmt: DatasetFormat) -> Self {
        self.format = fmt;
        self
    }

    /// Set split
    pub fn split(mut self, split: SplitType) -> Self {
        self.split = split;
        self
    }

    /// Set streaming mode
    pub fn streaming(mut self, mode: StreamingMode) -> Self {
        self.streaming = mode;
        self
    }

    /// Set number of workers
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set shuffle
    pub fn shuffle(mut self, enabled: bool) -> Self {
        self.shuffle = enabled;
        self
    }

    /// Set seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Dataset name cannot be empty".to_string());
        }
        if self.batch_size == 0 {
            return Err("Batch size must be > 0".to_string());
        }
        Ok(())
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Default)]
pub struct DatasetStats {
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns/features
    pub num_columns: usize,
    /// Size in bytes
    pub size_bytes: u64,
    /// Compression ratio (if applicable)
    pub compression_ratio: Option<f64>,
    /// Number of splits
    pub num_splits: usize,
}

impl DatasetStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Set rows
    pub fn rows(mut self, n: usize) -> Self {
        self.num_rows = n;
        self
    }

    /// Set columns
    pub fn columns(mut self, n: usize) -> Self {
        self.num_columns = n;
        self
    }

    /// Set size in bytes
    pub fn size_bytes(mut self, bytes: u64) -> Self {
        self.size_bytes = bytes;
        self
    }

    /// Set compression ratio
    pub fn compression_ratio(mut self, ratio: f64) -> Self {
        self.compression_ratio = Some(ratio);
        self
    }

    /// Set number of splits
    pub fn splits(mut self, n: usize) -> Self {
        self.num_splits = n;
        self
    }

    /// Format size as human readable
    pub fn format_size(&self) -> String {
        let bytes = self.size_bytes;
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Dataset: {} rows, {} cols, {}",
            self.num_rows, self.num_columns, self.format_size()
        )
    }
}

/// Download configuration
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Cache directory
    pub cache_dir: String,
    /// Force re-download
    pub force_download: bool,
    /// Resume partial downloads
    pub resume: bool,
    /// Verify checksums
    pub verify_checksums: bool,
    /// Maximum retries
    pub max_retries: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            cache_dir: String::from("~/.cache/datasets"),
            force_download: false,
            resume: true,
            verify_checksums: true,
            max_retries: 3,
        }
    }
}

impl DownloadConfig {
    /// Create new download config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cache directory
    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Set force download
    pub fn force_download(mut self, force: bool) -> Self {
        self.force_download = force;
        self
    }

    /// Set resume
    pub fn resume(mut self, enabled: bool) -> Self {
        self.resume = enabled;
        self
    }

    /// Set verify checksums
    pub fn verify_checksums(mut self, enabled: bool) -> Self {
        self.verify_checksums = enabled;
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }
}

/// Estimate download size based on format and row count
pub fn estimate_download_size(format: DatasetFormat, num_rows: usize, avg_row_bytes: usize) -> u64 {
    let raw_size = (num_rows * avg_row_bytes) as u64;
    
    // Apply format-specific compression estimates
    match format {
        DatasetFormat::Parquet => raw_size / 4, // ~4x compression
        DatasetFormat::Arrow => raw_size / 2,   // ~2x compression
        DatasetFormat::Csv | DatasetFormat::JsonLines => raw_size,
        DatasetFormat::Text => raw_size,
    }
}

/// Calculate dataset hash for cache invalidation
pub fn calculate_dataset_hash(name: &str, split: SplitType, config_hash: u64) -> u64 {
    let mut hash = 0u64;
    for b in name.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(b as u64);
    }
    hash = hash.wrapping_mul(31).wrapping_add(split as u64);
    hash = hash.wrapping_mul(31).wrapping_add(config_hash);
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_format_default() {
        assert_eq!(DatasetFormat::default(), DatasetFormat::Parquet);
    }

    #[test]
    fn test_dataset_format_as_str() {
        assert_eq!(DatasetFormat::Parquet.as_str(), "parquet");
        assert_eq!(DatasetFormat::Csv.as_str(), "csv");
        assert_eq!(DatasetFormat::JsonLines.as_str(), "jsonl");
        assert_eq!(DatasetFormat::Arrow.as_str(), "arrow");
        assert_eq!(DatasetFormat::Text.as_str(), "text");
    }

    #[test]
    fn test_dataset_format_extension() {
        assert_eq!(DatasetFormat::Parquet.extension(), ".parquet");
        assert_eq!(DatasetFormat::Csv.extension(), ".csv");
        assert_eq!(DatasetFormat::JsonLines.extension(), ".jsonl");
    }

    #[test]
    fn test_dataset_format_from_str() {
        assert_eq!(DatasetFormat::parse("parquet"), Some(DatasetFormat::Parquet));
        assert_eq!(DatasetFormat::parse("jsonlines"), Some(DatasetFormat::JsonLines));
        assert_eq!(DatasetFormat::parse("ipc"), Some(DatasetFormat::Arrow));
        assert_eq!(DatasetFormat::parse("txt"), Some(DatasetFormat::Text));
        assert_eq!(DatasetFormat::parse("unknown"), None);
    }

    #[test]
    fn test_dataset_format_list_all() {
        assert_eq!(DatasetFormat::list_all().len(), 5);
    }

    #[test]
    fn test_dataset_format_supports_compression() {
        assert!(DatasetFormat::Parquet.supports_compression());
        assert!(DatasetFormat::Arrow.supports_compression());
        assert!(!DatasetFormat::Csv.supports_compression());
        assert!(!DatasetFormat::JsonLines.supports_compression());
    }

    #[test]
    fn test_split_type_default() {
        assert_eq!(SplitType::default(), SplitType::Train);
    }

    #[test]
    fn test_split_type_as_str() {
        assert_eq!(SplitType::Train.as_str(), "train");
        assert_eq!(SplitType::Validation.as_str(), "validation");
        assert_eq!(SplitType::Test.as_str(), "test");
        assert_eq!(SplitType::All.as_str(), "all");
    }

    #[test]
    fn test_split_type_from_str() {
        assert_eq!(SplitType::parse("train"), Some(SplitType::Train));
        assert_eq!(SplitType::parse("val"), Some(SplitType::Validation));
        assert_eq!(SplitType::parse("dev"), Some(SplitType::Validation));
        assert_eq!(SplitType::parse("*"), Some(SplitType::All));
        assert_eq!(SplitType::parse("unknown"), None);
    }

    #[test]
    fn test_split_type_list_all() {
        assert_eq!(SplitType::list_all().len(), 4);
    }

    #[test]
    fn test_streaming_mode_default() {
        assert_eq!(StreamingMode::default(), StreamingMode::Disabled);
    }

    #[test]
    fn test_streaming_mode_as_str() {
        assert_eq!(StreamingMode::Disabled.as_str(), "disabled");
        assert_eq!(StreamingMode::Enabled.as_str(), "enabled");
        assert_eq!(StreamingMode::Auto.as_str(), "auto");
    }

    #[test]
    fn test_streaming_mode_from_str() {
        assert_eq!(StreamingMode::parse("off"), Some(StreamingMode::Disabled));
        assert_eq!(StreamingMode::parse("on"), Some(StreamingMode::Enabled));
        assert_eq!(StreamingMode::parse("true"), Some(StreamingMode::Enabled));
        assert_eq!(StreamingMode::parse("auto"), Some(StreamingMode::Auto));
    }

    #[test]
    fn test_streaming_mode_list_all() {
        assert_eq!(StreamingMode::list_all().len(), 3);
    }

    #[test]
    fn test_dataset_config_default() {
        let config = DatasetConfig::default();
        assert!(config.name.is_empty());
        assert_eq!(config.format, DatasetFormat::Parquet);
        assert_eq!(config.split, SplitType::Train);
        assert_eq!(config.batch_size, 32);
        assert!(config.shuffle);
    }

    #[test]
    fn test_dataset_config_builder() {
        let config = DatasetConfig::new("my-dataset")
            .format(DatasetFormat::Csv)
            .split(SplitType::Test)
            .streaming(StreamingMode::Enabled)
            .num_workers(8)
            .shuffle(false)
            .seed(42)
            .batch_size(64);

        assert_eq!(config.name, "my-dataset");
        assert_eq!(config.format, DatasetFormat::Csv);
        assert_eq!(config.split, SplitType::Test);
        assert_eq!(config.streaming, StreamingMode::Enabled);
        assert_eq!(config.num_workers, 8);
        assert!(!config.shuffle);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_dataset_config_validate() {
        let valid = DatasetConfig::new("dataset");
        assert!(valid.validate().is_ok());

        let no_name = DatasetConfig::default();
        assert!(no_name.validate().is_err());

        let zero_batch = DatasetConfig::new("ds").batch_size(0);
        assert!(zero_batch.validate().is_err());
    }

    #[test]
    fn test_dataset_stats_default() {
        let stats = DatasetStats::default();
        assert_eq!(stats.num_rows, 0);
        assert_eq!(stats.size_bytes, 0);
    }

    #[test]
    fn test_dataset_stats_builder() {
        let stats = DatasetStats::new()
            .rows(10000)
            .columns(10)
            .size_bytes(1024 * 1024)
            .compression_ratio(4.0)
            .splits(3);

        assert_eq!(stats.num_rows, 10000);
        assert_eq!(stats.num_columns, 10);
        assert_eq!(stats.size_bytes, 1024 * 1024);
        assert_eq!(stats.compression_ratio, Some(4.0));
        assert_eq!(stats.num_splits, 3);
    }

    #[test]
    fn test_dataset_stats_format_size() {
        assert_eq!(DatasetStats::new().size_bytes(500).format_size(), "500 B");
        assert_eq!(DatasetStats::new().size_bytes(2048).format_size(), "2.0 KB");
        assert_eq!(DatasetStats::new().size_bytes(1024 * 1024 * 5).format_size(), "5.0 MB");
        assert_eq!(DatasetStats::new().size_bytes(1024 * 1024 * 1024 * 2).format_size(), "2.00 GB");
    }

    #[test]
    fn test_dataset_stats_format() {
        let stats = DatasetStats::new().rows(1000).columns(5).size_bytes(10240);
        let formatted = stats.format();
        assert!(formatted.contains("1000 rows"));
        assert!(formatted.contains("5 cols"));
        assert!(formatted.contains("10.0 KB"));
    }

    #[test]
    fn test_download_config_default() {
        let config = DownloadConfig::default();
        assert!(config.cache_dir.contains("cache"));
        assert!(!config.force_download);
        assert!(config.resume);
        assert!(config.verify_checksums);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_download_config_builder() {
        let config = DownloadConfig::new()
            .cache_dir("/tmp/cache")
            .force_download(true)
            .resume(false)
            .verify_checksums(false)
            .max_retries(5);

        assert_eq!(config.cache_dir, "/tmp/cache");
        assert!(config.force_download);
        assert!(!config.resume);
        assert!(!config.verify_checksums);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_estimate_download_size() {
        let raw_rows = 1000;
        let row_size = 1000;

        let parquet_size = estimate_download_size(DatasetFormat::Parquet, raw_rows, row_size);
        let csv_size = estimate_download_size(DatasetFormat::Csv, raw_rows, row_size);

        assert!(parquet_size < csv_size);
        assert_eq!(csv_size, 1_000_000);
        assert_eq!(parquet_size, 250_000);
    }

    #[test]
    fn test_calculate_dataset_hash() {
        let hash1 = calculate_dataset_hash("dataset1", SplitType::Train, 123);
        let hash2 = calculate_dataset_hash("dataset1", SplitType::Train, 123);
        let hash3 = calculate_dataset_hash("dataset2", SplitType::Train, 123);
        let hash4 = calculate_dataset_hash("dataset1", SplitType::Test, 123);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_ne!(hash1, hash4);
    }

    #[test]
    fn test_calculate_dataset_hash_deterministic() {
        let hash = calculate_dataset_hash("test", SplitType::Train, 0);
        assert!(hash > 0);
    }
}
