# Deployment API

## Module: `batuta_ground_truth_mlops_corpus::deployment`

### QuantizationConfig

```rust
pub struct QuantizationConfig {
    pub quantization_type: QuantizationType,
    pub per_channel: bool,
    pub symmetric: bool,
    pub calibration_samples: usize,
}

impl QuantizationConfig {
    pub fn new() -> Self;
    pub fn quantization_type(self, qt: QuantizationType) -> Self;
    pub fn per_channel(self, v: bool) -> Self;
    pub fn symmetric(self, v: bool) -> Self;
}
```

### QuantizationType

```rust
pub enum QuantizationType {
    Int8,     // 8x compression
    Int4,     // 16x compression
    Float16,  // 4x compression
    Dynamic,  // Dynamic quantization
}
```

### ExportConfig

```rust
pub struct ExportConfig {
    pub format: ExportFormat,
    pub output_path: Option<PathBuf>,
    pub include_metadata: bool,
    pub compress: bool,
    pub compression_level: u8,
}

impl ExportConfig {
    pub fn new() -> Self;
    pub fn format(self, fmt: ExportFormat) -> Self;
    pub fn output_path(self, path: PathBuf) -> Self;
    pub fn compress(self, v: bool) -> Self;
}
```

### ExportFormat

```rust
pub enum ExportFormat {
    Apr,       // Sovereign AI Stack native format
    Json,      // JSON format
    Binary,    // Binary format
    OnnxMeta,  // ONNX metadata only
}
```

### Quantizer

```rust
pub struct Quantizer { ... }

impl Quantizer {
    pub fn new(config: QuantizationConfig) -> Self;
    pub fn quantize(&self, weights: &[f64]) -> QuantizedModel;
    pub fn compression_ratio(&self) -> f64;
}
```
