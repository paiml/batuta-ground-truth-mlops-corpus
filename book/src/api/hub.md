# Hub API

## Module: `batuta_ground_truth_mlops_corpus::hub`

### RegistryConfig

```rust
pub struct RegistryConfig {
    pub name: String,
    pub namespace: String,
    pub versioning_scheme: VersioningScheme,
}

impl RegistryConfig {
    pub fn new(name: &str) -> Self;
    pub fn namespace(self, ns: &str) -> Self;
    pub fn versioning_scheme(self, vs: VersioningScheme) -> Self;
    pub fn validate(&self) -> Result<(), String>;
}
```

### ModelStage

```rust
pub enum ModelStage {
    Development,
    Staging,
    Production,
    Archived,
}
```

### parse_version

```rust
pub fn parse_version(version: &str) -> Option<VersionInfo>;

pub struct VersionInfo {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub prerelease: Option<String>,
}
```

### DatasetConfig

```rust
pub struct DatasetConfig {
    pub name: String,
    pub format: DatasetFormat,
    pub split: SplitType,
    pub batch_size: usize,
}

impl DatasetConfig {
    pub fn new(name: &str) -> Self;
    pub fn format(self, fmt: DatasetFormat) -> Self;
    pub fn split(self, s: SplitType) -> Self;
}
```

### DatasetFormat

```rust
pub enum DatasetFormat {
    Parquet,
    Csv,
    JsonLines,
    Arrow,
    Text,
}
```
