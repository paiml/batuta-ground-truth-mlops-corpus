//! Hub Module
//!
//! Model registry, versioning, and dataset management utilities.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.hub import (
//!     register_model, ModelStage, create_registry_config,
//!     parse_version, DatasetFormat
//! )
//! ```
//!
//! # Example
//!
//! ```rust
//! use batuta_ground_truth_mlops_corpus::hub::{
//!     RegistryConfig, ModelStage, VersioningScheme,
//!     parse_version, DatasetConfig, DatasetFormat
//! };
//!
//! // Model registry
//! let config = RegistryConfig::new("my-model")
//!     .namespace("org")
//!     .versioning_scheme(VersioningScheme::Semantic);
//! assert!(config.validate().is_ok());
//!
//! // Version parsing
//! let version = parse_version("1.2.3-beta").unwrap();
//! assert_eq!(version.major, 1);
//!
//! // Dataset config
//! let dataset = DatasetConfig::new("imdb")
//!     .format(DatasetFormat::Parquet);
//! ```

pub mod registry;
pub mod versioning;
pub mod datasets;

pub use registry::{
    ModelStage,
    ModelVersion,
    RegistryConfig,
    RegistryStats,
    VersioningScheme,
    compare_versions,
    increment_version,
};

pub use versioning::{
    ChangeType,
    DiffType,
    ModelDiff,
    VersionInfo,
    VersionStats,
    parse_version,
};

pub use datasets::{
    DatasetConfig,
    DatasetFormat,
    DatasetStats,
    DownloadConfig,
    SplitType,
    StreamingMode,
    calculate_dataset_hash,
    estimate_download_size,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_module_exports() {
        // Verify all types are accessible
        let _ = ModelStage::default();
        let _ = VersioningScheme::default();
        let _ = DatasetFormat::default();
        let _ = ChangeType::default();
    }

    #[test]
    fn test_hub_integration() {
        // Registry + versioning integration
        let config = RegistryConfig::new("test-model")
            .versioning_scheme(VersioningScheme::Semantic);
        
        let version = ModelVersion::new("1.0.0")
            .stage(ModelStage::Development);
        
        assert!(config.validate().is_ok());
        assert!(version.validate().is_ok());
    }

    #[test]
    fn test_version_parsing_integration() {
        let info = parse_version("2.1.0-rc1").unwrap();
        let bumped = info.bump(ChangeType::Feature);
        assert_eq!(bumped.format(), "2.2.0");
    }

    #[test]
    fn test_dataset_integration() {
        let config = DatasetConfig::new("squad")
            .format(DatasetFormat::Parquet)
            .split(SplitType::Train);
        
        let hash = calculate_dataset_hash(&config.name, config.split, 0);
        assert!(hash > 0);
    }
}
