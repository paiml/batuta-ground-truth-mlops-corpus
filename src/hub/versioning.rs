//! Version Management
//!
//! Provides semantic versioning, comparison, and diff utilities.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.hub import parse_version, create_version_info, ChangeType
//! version = parse_version("1.2.3")
//! ```

/// Change type for version bumps
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChangeType {
    /// Breaking change (major bump)
    Breaking,
    /// New feature (minor bump)
    Feature,
    /// Bug fix (patch bump)
    #[default]
    Fix,
    /// Documentation only
    Docs,
    /// Internal refactor
    Refactor,
}

impl ChangeType {
    /// Get change type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Breaking => "breaking",
            Self::Feature => "feature",
            Self::Fix => "fix",
            Self::Docs => "docs",
            Self::Refactor => "refactor",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "breaking" | "major" => Some(Self::Breaking),
            "feature" | "minor" | "feat" => Some(Self::Feature),
            "fix" | "patch" | "bugfix" => Some(Self::Fix),
            "docs" | "documentation" => Some(Self::Docs),
            "refactor" | "refactoring" => Some(Self::Refactor),
            _ => None,
        }
    }

    /// List all change types
    pub fn list_all() -> Vec<Self> {
        vec![Self::Breaking, Self::Feature, Self::Fix, Self::Docs, Self::Refactor]
    }

    /// Get the version bump type
    pub fn bump_type(&self) -> &'static str {
        match self {
            Self::Breaking => "major",
            Self::Feature => "minor",
            Self::Fix | Self::Docs | Self::Refactor => "patch",
        }
    }
}

/// Diff type for comparing versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiffType {
    /// No changes
    #[default]
    None,
    /// Parameter changes
    Parameters,
    /// Architecture changes
    Architecture,
    /// Weight changes only
    Weights,
    /// Config changes only
    Config,
}

impl DiffType {
    /// Get diff type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Parameters => "parameters",
            Self::Architecture => "architecture",
            Self::Weights => "weights",
            Self::Config => "config",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "parameters" | "params" => Some(Self::Parameters),
            "architecture" | "arch" => Some(Self::Architecture),
            "weights" => Some(Self::Weights),
            "config" | "configuration" => Some(Self::Config),
            _ => None,
        }
    }

    /// List all diff types
    pub fn list_all() -> Vec<Self> {
        vec![Self::None, Self::Parameters, Self::Architecture, Self::Weights, Self::Config]
    }
}

/// Parsed version information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionInfo {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Pre-release tag (e.g., "alpha", "beta", "rc1")
    pub prerelease: Option<String>,
    /// Build metadata
    pub build_metadata: Option<String>,
}

impl Default for VersionInfo {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 1,
            patch: 0,
            prerelease: None,
            build_metadata: None,
        }
    }
}

impl VersionInfo {
    /// Create new version info
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            ..Default::default()
        }
    }

    /// Set prerelease tag
    pub fn prerelease(mut self, tag: impl Into<String>) -> Self {
        self.prerelease = Some(tag.into());
        self
    }

    /// Set build metadata
    pub fn build_metadata(mut self, meta: impl Into<String>) -> Self {
        self.build_metadata = Some(meta.into());
        self
    }

    /// Format as version string
    pub fn format(&self) -> String {
        let mut s = format!("{}.{}.{}", self.major, self.minor, self.patch);
        if let Some(ref pre) = self.prerelease {
            s.push('-');
            s.push_str(pre);
        }
        if let Some(ref meta) = self.build_metadata {
            s.push('+');
            s.push_str(meta);
        }
        s
    }

    /// Check if this is a prerelease version
    pub fn is_prerelease(&self) -> bool {
        self.prerelease.is_some()
    }

    /// Compare with another version (ignoring prerelease/build)
    pub fn compare_core(&self, other: &Self) -> std::cmp::Ordering {
        match self.major.cmp(&other.major) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match self.minor.cmp(&other.minor) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        self.patch.cmp(&other.patch)
    }

    /// Bump version by change type
    pub fn bump(&self, change: ChangeType) -> Self {
        match change {
            ChangeType::Breaking => Self::new(self.major + 1, 0, 0),
            ChangeType::Feature => Self::new(self.major, self.minor + 1, 0),
            ChangeType::Fix | ChangeType::Docs | ChangeType::Refactor => {
                Self::new(self.major, self.minor, self.patch + 1)
            }
        }
    }
}

/// Parse a version string into VersionInfo
pub fn parse_version(version: &str) -> Option<VersionInfo> {
    let version = version.trim().trim_start_matches('v');
    
    // Split off build metadata
    let (version, build_meta) = match version.split_once('+') {
        Some((v, m)) => (v, Some(m.to_string())),
        None => (version, None),
    };

    // Split off prerelease
    let (version, prerelease) = match version.split_once('-') {
        Some((v, p)) => (v, Some(p.to_string())),
        None => (version, None),
    };

    let parts: Vec<&str> = version.split('.').collect();
    let major = parts.first()?.parse().ok()?;
    let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    Some(VersionInfo {
        major,
        minor,
        patch,
        prerelease,
        build_metadata: build_meta,
    })
}

/// Model diff result
#[derive(Debug, Clone, Default)]
pub struct ModelDiff {
    /// Type of diff
    pub diff_type: DiffType,
    /// Number of changed parameters
    pub changed_params: usize,
    /// Parameter delta (positive = more params)
    pub param_delta: i64,
    /// List of changed layers/components
    pub changed_components: Vec<String>,
    /// Summary description
    pub summary: String,
}

impl ModelDiff {
    /// Create new model diff
    pub fn new(diff_type: DiffType) -> Self {
        Self {
            diff_type,
            ..Default::default()
        }
    }

    /// Set changed params
    pub fn changed_params(mut self, count: usize) -> Self {
        self.changed_params = count;
        self
    }

    /// Set param delta
    pub fn param_delta(mut self, delta: i64) -> Self {
        self.param_delta = delta;
        self
    }

    /// Add changed component
    pub fn add_component(mut self, component: impl Into<String>) -> Self {
        self.changed_components.push(component.into());
        self
    }

    /// Set summary
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Format the diff
    pub fn format(&self) -> String {
        let mut s = format!("Diff type: {}\n", self.diff_type.as_str());
        if self.changed_params > 0 {
            s.push_str(&format!("Changed params: {}\n", self.changed_params));
        }
        if self.param_delta != 0 {
            s.push_str(&format!("Param delta: {:+}\n", self.param_delta));
        }
        if !self.changed_components.is_empty() {
            s.push_str(&format!("Components: {}\n", self.changed_components.join(", ")));
        }
        if !self.summary.is_empty() {
            s.push_str(&format!("Summary: {}", self.summary));
        }
        s
    }
}

/// Version statistics
#[derive(Debug, Clone, Default)]
pub struct VersionStats {
    /// Total versions
    pub total_versions: usize,
    /// Major releases
    pub major_releases: usize,
    /// Minor releases
    pub minor_releases: usize,
    /// Patch releases
    pub patch_releases: usize,
    /// Prerelease versions
    pub prereleases: usize,
}

impl VersionStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total versions
    pub fn total(mut self, n: usize) -> Self {
        self.total_versions = n;
        self
    }

    /// Set major releases
    pub fn majors(mut self, n: usize) -> Self {
        self.major_releases = n;
        self
    }

    /// Set minor releases
    pub fn minors(mut self, n: usize) -> Self {
        self.minor_releases = n;
        self
    }

    /// Set patch releases
    pub fn patches(mut self, n: usize) -> Self {
        self.patch_releases = n;
        self
    }

    /// Set prereleases
    pub fn prereleases(mut self, n: usize) -> Self {
        self.prereleases = n;
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Versions: {} total ({} major, {} minor, {} patch, {} pre)",
            self.total_versions,
            self.major_releases,
            self.minor_releases,
            self.patch_releases,
            self.prereleases
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_type_default() {
        let ct = ChangeType::default();
        assert_eq!(ct, ChangeType::Fix);
    }

    #[test]
    fn test_change_type_as_str() {
        assert_eq!(ChangeType::Breaking.as_str(), "breaking");
        assert_eq!(ChangeType::Feature.as_str(), "feature");
        assert_eq!(ChangeType::Fix.as_str(), "fix");
        assert_eq!(ChangeType::Docs.as_str(), "docs");
        assert_eq!(ChangeType::Refactor.as_str(), "refactor");
    }

    #[test]
    fn test_change_type_from_str() {
        assert_eq!(ChangeType::parse("major"), Some(ChangeType::Breaking));
        assert_eq!(ChangeType::parse("feat"), Some(ChangeType::Feature));
        assert_eq!(ChangeType::parse("bugfix"), Some(ChangeType::Fix));
        assert_eq!(ChangeType::parse("documentation"), Some(ChangeType::Docs));
        assert_eq!(ChangeType::parse("refactoring"), Some(ChangeType::Refactor));
        assert_eq!(ChangeType::parse("unknown"), None);
    }

    #[test]
    fn test_change_type_list_all() {
        assert_eq!(ChangeType::list_all().len(), 5);
    }

    #[test]
    fn test_change_type_bump_type() {
        assert_eq!(ChangeType::Breaking.bump_type(), "major");
        assert_eq!(ChangeType::Feature.bump_type(), "minor");
        assert_eq!(ChangeType::Fix.bump_type(), "patch");
        assert_eq!(ChangeType::Docs.bump_type(), "patch");
        assert_eq!(ChangeType::Refactor.bump_type(), "patch");
    }

    #[test]
    fn test_diff_type_default() {
        assert_eq!(DiffType::default(), DiffType::None);
    }

    #[test]
    fn test_diff_type_as_str() {
        assert_eq!(DiffType::None.as_str(), "none");
        assert_eq!(DiffType::Parameters.as_str(), "parameters");
        assert_eq!(DiffType::Architecture.as_str(), "architecture");
        assert_eq!(DiffType::Weights.as_str(), "weights");
        assert_eq!(DiffType::Config.as_str(), "config");
    }

    #[test]
    fn test_diff_type_from_str() {
        assert_eq!(DiffType::parse("params"), Some(DiffType::Parameters));
        assert_eq!(DiffType::parse("arch"), Some(DiffType::Architecture));
        assert_eq!(DiffType::parse("configuration"), Some(DiffType::Config));
        assert_eq!(DiffType::parse("invalid"), None);
    }

    #[test]
    fn test_diff_type_list_all() {
        assert_eq!(DiffType::list_all().len(), 5);
    }

    #[test]
    fn test_version_info_default() {
        let v = VersionInfo::default();
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
        assert!(v.prerelease.is_none());
    }

    #[test]
    fn test_version_info_new() {
        let v = VersionInfo::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_info_builder() {
        let v = VersionInfo::new(1, 0, 0)
            .prerelease("alpha")
            .build_metadata("20240101");
        
        assert_eq!(v.prerelease, Some("alpha".to_string()));
        assert_eq!(v.build_metadata, Some("20240101".to_string()));
    }

    #[test]
    fn test_version_info_format() {
        assert_eq!(VersionInfo::new(1, 2, 3).format(), "1.2.3");
        assert_eq!(VersionInfo::new(1, 0, 0).prerelease("beta").format(), "1.0.0-beta");
        assert_eq!(
            VersionInfo::new(2, 0, 0).build_metadata("abc123").format(),
            "2.0.0+abc123"
        );
        assert_eq!(
            VersionInfo::new(1, 0, 0).prerelease("rc1").build_metadata("build").format(),
            "1.0.0-rc1+build"
        );
    }

    #[test]
    fn test_version_info_is_prerelease() {
        assert!(!VersionInfo::new(1, 0, 0).is_prerelease());
        assert!(VersionInfo::new(1, 0, 0).prerelease("alpha").is_prerelease());
    }

    #[test]
    fn test_version_info_compare_core() {
        let v1 = VersionInfo::new(1, 0, 0);
        let v2 = VersionInfo::new(1, 0, 0);
        let v3 = VersionInfo::new(1, 0, 1);
        let v4 = VersionInfo::new(2, 0, 0);

        assert_eq!(v1.compare_core(&v2), std::cmp::Ordering::Equal);
        assert_eq!(v1.compare_core(&v3), std::cmp::Ordering::Less);
        assert_eq!(v4.compare_core(&v1), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_version_info_bump() {
        let v = VersionInfo::new(1, 2, 3);
        
        let major = v.bump(ChangeType::Breaking);
        assert_eq!(major, VersionInfo::new(2, 0, 0));
        
        let minor = v.bump(ChangeType::Feature);
        assert_eq!(minor, VersionInfo::new(1, 3, 0));
        
        let patch = v.bump(ChangeType::Fix);
        assert_eq!(patch, VersionInfo::new(1, 2, 4));
    }

    #[test]
    fn test_parse_version_basic() {
        let v = parse_version("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_parse_version_with_v_prefix() {
        let v = parse_version("v2.0.0").unwrap();
        assert_eq!(v.major, 2);
    }

    #[test]
    fn test_parse_version_partial() {
        let v1 = parse_version("1").unwrap();
        assert_eq!(v1, VersionInfo::new(1, 0, 0));

        let v2 = parse_version("1.2").unwrap();
        assert_eq!(v2, VersionInfo::new(1, 2, 0));
    }

    #[test]
    fn test_parse_version_with_prerelease() {
        let v = parse_version("1.0.0-alpha.1").unwrap();
        assert_eq!(v.prerelease, Some("alpha.1".to_string()));
    }

    #[test]
    fn test_parse_version_with_build_metadata() {
        let v = parse_version("1.0.0+build.123").unwrap();
        assert_eq!(v.build_metadata, Some("build.123".to_string()));
    }

    #[test]
    fn test_parse_version_full() {
        let v = parse_version("1.0.0-rc1+20240101").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.prerelease, Some("rc1".to_string()));
        assert_eq!(v.build_metadata, Some("20240101".to_string()));
    }

    #[test]
    fn test_parse_version_invalid() {
        assert!(parse_version("").is_none());
        assert!(parse_version("abc").is_none());
    }

    #[test]
    fn test_model_diff_default() {
        let diff = ModelDiff::default();
        assert_eq!(diff.diff_type, DiffType::None);
        assert_eq!(diff.changed_params, 0);
    }

    #[test]
    fn test_model_diff_builder() {
        let diff = ModelDiff::new(DiffType::Parameters)
            .changed_params(1000)
            .param_delta(-500)
            .add_component("layer1")
            .add_component("layer2")
            .summary("Reduced model size");

        assert_eq!(diff.diff_type, DiffType::Parameters);
        assert_eq!(diff.changed_params, 1000);
        assert_eq!(diff.param_delta, -500);
        assert_eq!(diff.changed_components.len(), 2);
        assert_eq!(diff.summary, "Reduced model size");
    }

    #[test]
    fn test_model_diff_format() {
        let diff = ModelDiff::new(DiffType::Weights)
            .changed_params(100)
            .param_delta(50);
        
        let formatted = diff.format();
        assert!(formatted.contains("weights"));
        assert!(formatted.contains("100"));
        assert!(formatted.contains("+50"));
    }

    #[test]
    fn test_version_stats_default() {
        let stats = VersionStats::default();
        assert_eq!(stats.total_versions, 0);
    }

    #[test]
    fn test_version_stats_builder() {
        let stats = VersionStats::new()
            .total(10)
            .majors(2)
            .minors(5)
            .patches(3)
            .prereleases(1);

        assert_eq!(stats.total_versions, 10);
        assert_eq!(stats.major_releases, 2);
        assert_eq!(stats.minor_releases, 5);
        assert_eq!(stats.patch_releases, 3);
        assert_eq!(stats.prereleases, 1);
    }

    #[test]
    fn test_version_stats_format() {
        let stats = VersionStats::new().total(5).majors(1).minors(2).patches(2);
        let formatted = stats.format();
        assert!(formatted.contains("5 total"));
        assert!(formatted.contains("1 major"));
    }
}
