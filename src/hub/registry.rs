//! Model Registry
//!
//! Provides model registration, versioning, and stage management.
//!
//! # Python Equivalent (MLflow/HuggingFace Hub)
//! ```python
//! from hf_gtc.hub import register_model, ModelStage, create_registry_config
//! config = create_registry_config(name="my-model", namespace="org")
//! register_model(model, config)
//! ```

use std::collections::HashMap;

/// Model lifecycle stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelStage {
    /// Development stage
    #[default]
    Development,
    /// Staging for testing
    Staging,
    /// Production ready
    Production,
    /// Archived/deprecated
    Archived,
}

impl ModelStage {
    /// Get stage name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Staging => "staging",
            Self::Production => "production",
            Self::Archived => "archived",
        }
    }

    /// Parse stage from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "development" | "dev" => Some(Self::Development),
            "staging" | "stage" => Some(Self::Staging),
            "production" | "prod" => Some(Self::Production),
            "archived" | "archive" => Some(Self::Archived),
            _ => None,
        }
    }

    /// List all valid stages
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Development,
            Self::Staging,
            Self::Production,
            Self::Archived,
        ]
    }

    /// Check if stage can transition to another
    pub fn can_transition_to(&self, target: Self) -> bool {
        match (self, target) {
            (Self::Development, Self::Staging) => true,
            (Self::Staging, Self::Production) => true,
            (Self::Staging, Self::Development) => true,
            (Self::Production, Self::Archived) => true,
            (Self::Production, Self::Staging) => true,
            (Self::Archived, _) => false,
            _ => false,
        }
    }
}

/// Versioning scheme for models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VersioningScheme {
    /// Semantic versioning (major.minor.patch)
    #[default]
    Semantic,
    /// Date-based versioning (YYYY.MM.DD)
    DateBased,
    /// Sequential integer versioning
    Sequential,
    /// Git commit hash based
    GitHash,
}

impl VersioningScheme {
    /// Get scheme name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::DateBased => "date_based",
            Self::Sequential => "sequential",
            Self::GitHash => "git_hash",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "semantic" | "semver" => Some(Self::Semantic),
            "date_based" | "date" => Some(Self::DateBased),
            "sequential" | "seq" => Some(Self::Sequential),
            "git_hash" | "git" => Some(Self::GitHash),
            _ => None,
        }
    }

    /// List all schemes
    pub fn list_all() -> Vec<Self> {
        vec![Self::Semantic, Self::DateBased, Self::Sequential, Self::GitHash]
    }
}

/// Configuration for model registry
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Model name
    pub name: String,
    /// Namespace/organization
    pub namespace: String,
    /// Versioning scheme
    pub versioning_scheme: VersioningScheme,
    /// Enable automatic staging
    pub auto_staging: bool,
    /// Require approval for production
    pub require_approval: bool,
    /// Custom tags
    pub tags: HashMap<String, String>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            namespace: String::from("default"),
            versioning_scheme: VersioningScheme::Semantic,
            auto_staging: false,
            require_approval: true,
            tags: HashMap::new(),
        }
    }
}

impl RegistryConfig {
    /// Create new registry config
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set namespace
    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Set versioning scheme
    pub fn versioning_scheme(mut self, scheme: VersioningScheme) -> Self {
        self.versioning_scheme = scheme;
        self
    }

    /// Enable auto staging
    pub fn auto_staging(mut self, enabled: bool) -> Self {
        self.auto_staging = enabled;
        self
    }

    /// Set require approval
    pub fn require_approval(mut self, required: bool) -> Self {
        self.require_approval = required;
        self
    }

    /// Add a tag
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }
        if self.name.len() > 256 {
            return Err("Model name too long (max 256 chars)".to_string());
        }
        if self.namespace.is_empty() {
            return Err("Namespace cannot be empty".to_string());
        }
        Ok(())
    }

    /// Get full model identifier
    pub fn full_name(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }
}

/// Registered model version
#[derive(Debug, Clone)]
pub struct ModelVersion {
    /// Version string
    pub version: String,
    /// Current stage
    pub stage: ModelStage,
    /// Description
    pub description: String,
    /// Metrics associated with this version
    pub metrics: HashMap<String, f64>,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
    /// Model artifact path/URI
    pub artifact_uri: String,
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self {
            version: String::from("0.1.0"),
            stage: ModelStage::Development,
            description: String::new(),
            metrics: HashMap::new(),
            created_at: 0,
            artifact_uri: String::new(),
        }
    }
}

impl ModelVersion {
    /// Create new model version
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            ..Default::default()
        }
    }

    /// Set stage
    pub fn stage(mut self, stage: ModelStage) -> Self {
        self.stage = stage;
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add metric
    pub fn metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Set artifact URI
    pub fn artifact_uri(mut self, uri: impl Into<String>) -> Self {
        self.artifact_uri = uri.into();
        self
    }

    /// Set creation timestamp
    pub fn created_at(mut self, ts: u64) -> Self {
        self.created_at = ts;
        self
    }

    /// Validate version
    pub fn validate(&self) -> Result<(), String> {
        if self.version.is_empty() {
            return Err("Version cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Registry statistics
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    /// Total models
    pub total_models: usize,
    /// Total versions
    pub total_versions: usize,
    /// Models by stage
    pub by_stage: HashMap<String, usize>,
}

impl RegistryStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Add model count
    pub fn with_models(mut self, count: usize) -> Self {
        self.total_models = count;
        self
    }

    /// Add version count
    pub fn with_versions(mut self, count: usize) -> Self {
        self.total_versions = count;
        self
    }

    /// Add stage count
    pub fn with_stage_count(mut self, stage: ModelStage, count: usize) -> Self {
        self.by_stage.insert(stage.as_str().to_string(), count);
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Registry: {} models, {} versions",
            self.total_models, self.total_versions
        )
    }
}

/// Compare two versions semantically
pub fn compare_versions(v1: &str, v2: &str) -> std::cmp::Ordering {
    let parse = |v: &str| -> Vec<u32> {
        v.split('.')
            .filter_map(|s| s.parse().ok())
            .collect()
    };

    let parts1 = parse(v1);
    let parts2 = parse(v2);

    for (a, b) in parts1.iter().zip(parts2.iter()) {
        match a.cmp(b) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }

    parts1.len().cmp(&parts2.len())
}

/// Increment a semantic version
pub fn increment_version(version: &str, bump_type: &str) -> String {
    let parts: Vec<u32> = version
        .split('.')
        .filter_map(|s| s.parse().ok())
        .collect();

    let (major, minor, patch) = match parts.as_slice() {
        [m, n, p, ..] => (*m, *n, *p),
        [m, n] => (*m, *n, 0),
        [m] => (*m, 0, 0),
        [] => (0, 0, 0),
    };

    match bump_type {
        "major" => format!("{}.0.0", major + 1),
        "minor" => format!("{}.{}.0", major, minor + 1),
        _ => format!("{}.{}.{}", major, minor, patch + 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_stage_default() {
        let stage = ModelStage::default();
        assert_eq!(stage, ModelStage::Development);
    }

    #[test]
    fn test_model_stage_as_str() {
        assert_eq!(ModelStage::Development.as_str(), "development");
        assert_eq!(ModelStage::Staging.as_str(), "staging");
        assert_eq!(ModelStage::Production.as_str(), "production");
        assert_eq!(ModelStage::Archived.as_str(), "archived");
    }

    #[test]
    fn test_model_stage_from_str() {
        assert_eq!(ModelStage::parse("dev"), Some(ModelStage::Development));
        assert_eq!(ModelStage::parse("staging"), Some(ModelStage::Staging));
        assert_eq!(ModelStage::parse("prod"), Some(ModelStage::Production));
        assert_eq!(ModelStage::parse("archived"), Some(ModelStage::Archived));
        assert_eq!(ModelStage::parse("unknown"), None);
    }

    #[test]
    fn test_model_stage_list_all() {
        let stages = ModelStage::list_all();
        assert_eq!(stages.len(), 4);
    }

    #[test]
    fn test_model_stage_transitions() {
        assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Development));
        assert!(ModelStage::Production.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Production.can_transition_to(ModelStage::Staging));
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Production));
        assert!(!ModelStage::Development.can_transition_to(ModelStage::Production));
    }

    #[test]
    fn test_versioning_scheme_default() {
        let scheme = VersioningScheme::default();
        assert_eq!(scheme, VersioningScheme::Semantic);
    }

    #[test]
    fn test_versioning_scheme_as_str() {
        assert_eq!(VersioningScheme::Semantic.as_str(), "semantic");
        assert_eq!(VersioningScheme::DateBased.as_str(), "date_based");
        assert_eq!(VersioningScheme::Sequential.as_str(), "sequential");
        assert_eq!(VersioningScheme::GitHash.as_str(), "git_hash");
    }

    #[test]
    fn test_versioning_scheme_from_str() {
        assert_eq!(VersioningScheme::parse("semver"), Some(VersioningScheme::Semantic));
        assert_eq!(VersioningScheme::parse("date"), Some(VersioningScheme::DateBased));
        assert_eq!(VersioningScheme::parse("seq"), Some(VersioningScheme::Sequential));
        assert_eq!(VersioningScheme::parse("git"), Some(VersioningScheme::GitHash));
        assert_eq!(VersioningScheme::parse("unknown"), None);
    }

    #[test]
    fn test_versioning_scheme_list_all() {
        let schemes = VersioningScheme::list_all();
        assert_eq!(schemes.len(), 4);
    }

    #[test]
    fn test_registry_config_default() {
        let config = RegistryConfig::default();
        assert!(config.name.is_empty());
        assert_eq!(config.namespace, "default");
        assert_eq!(config.versioning_scheme, VersioningScheme::Semantic);
        assert!(!config.auto_staging);
        assert!(config.require_approval);
    }

    #[test]
    fn test_registry_config_builder() {
        let config = RegistryConfig::new("my-model")
            .namespace("org")
            .versioning_scheme(VersioningScheme::DateBased)
            .auto_staging(true)
            .require_approval(false)
            .tag("env", "test");

        assert_eq!(config.name, "my-model");
        assert_eq!(config.namespace, "org");
        assert_eq!(config.versioning_scheme, VersioningScheme::DateBased);
        assert!(config.auto_staging);
        assert!(!config.require_approval);
        assert_eq!(config.tags.get("env"), Some(&"test".to_string()));
    }

    #[test]
    fn test_registry_config_validate() {
        let valid = RegistryConfig::new("model").namespace("ns");
        assert!(valid.validate().is_ok());

        let no_name = RegistryConfig::default();
        assert!(no_name.validate().is_err());

        let long_name = RegistryConfig::new("a".repeat(300)).namespace("ns");
        assert!(long_name.validate().is_err());

        let no_ns = RegistryConfig::new("model").namespace("");
        assert!(no_ns.validate().is_err());
    }

    #[test]
    fn test_registry_config_full_name() {
        let config = RegistryConfig::new("my-model").namespace("org");
        assert_eq!(config.full_name(), "org/my-model");
    }

    #[test]
    fn test_model_version_default() {
        let version = ModelVersion::default();
        assert_eq!(version.version, "0.1.0");
        assert_eq!(version.stage, ModelStage::Development);
        assert!(version.description.is_empty());
    }

    #[test]
    fn test_model_version_builder() {
        let version = ModelVersion::new("1.0.0")
            .stage(ModelStage::Production)
            .description("Initial release")
            .metric("accuracy", 0.95)
            .artifact_uri("s3://bucket/model")
            .created_at(1234567890);

        assert_eq!(version.version, "1.0.0");
        assert_eq!(version.stage, ModelStage::Production);
        assert_eq!(version.description, "Initial release");
        assert_eq!(version.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(version.artifact_uri, "s3://bucket/model");
        assert_eq!(version.created_at, 1234567890);
    }

    #[test]
    fn test_model_version_validate() {
        let valid = ModelVersion::new("1.0.0");
        assert!(valid.validate().is_ok());

        let invalid = ModelVersion::new("");
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_registry_stats_default() {
        let stats = RegistryStats::default();
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.total_versions, 0);
    }

    #[test]
    fn test_registry_stats_builder() {
        let stats = RegistryStats::new()
            .with_models(10)
            .with_versions(50)
            .with_stage_count(ModelStage::Production, 5);

        assert_eq!(stats.total_models, 10);
        assert_eq!(stats.total_versions, 50);
        assert_eq!(stats.by_stage.get("production"), Some(&5));
    }

    #[test]
    fn test_registry_stats_format() {
        let stats = RegistryStats::new().with_models(5).with_versions(20);
        assert!(stats.format().contains("5 models"));
        assert!(stats.format().contains("20 versions"));
    }

    #[test]
    fn test_compare_versions() {
        assert_eq!(compare_versions("1.0.0", "1.0.0"), std::cmp::Ordering::Equal);
        assert_eq!(compare_versions("1.0.0", "1.0.1"), std::cmp::Ordering::Less);
        assert_eq!(compare_versions("1.1.0", "1.0.1"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("2.0.0", "1.9.9"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("1.0", "1.0.0"), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_increment_version() {
        assert_eq!(increment_version("1.0.0", "patch"), "1.0.1");
        assert_eq!(increment_version("1.0.0", "minor"), "1.1.0");
        assert_eq!(increment_version("1.0.0", "major"), "2.0.0");
        assert_eq!(increment_version("1.2", "patch"), "1.2.1");
        assert_eq!(increment_version("1", "minor"), "1.1.0");
    }

    #[test]
    fn test_increment_version_edge_cases() {
        assert_eq!(increment_version("", "patch"), "0.0.1");
        assert_eq!(increment_version("invalid", "patch"), "0.0.1");
    }
}
