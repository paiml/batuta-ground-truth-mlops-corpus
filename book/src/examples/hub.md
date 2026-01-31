# Model Hub & Deployment Example

Run with: `cargo run --example hub_demo`

## Overview

Demonstrates model registry and deployment utilities:
- Model registry configuration
- Version management
- Dataset configuration
- Quantization
- Export formats

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::hub::{
    RegistryConfig, ModelStage, VersioningScheme,
    parse_version, DatasetConfig, DatasetFormat,
};
use batuta_ground_truth_mlops_corpus::deployment::{
    QuantizationConfig, QuantizationType,
    ExportConfig, ExportFormat,
};

fn main() {
    // Registry configuration
    let registry = RegistryConfig::new("sentiment-classifier")
        .namespace("paiml")
        .versioning_scheme(VersioningScheme::Semantic);

    // Version parsing
    if let Some(ver) = parse_version("1.2.3-beta") {
        println!("major: {}, minor: {}, patch: {}",
            ver.major, ver.minor, ver.patch);
    }

    // Dataset configuration
    let dataset = DatasetConfig::new("imdb-reviews")
        .format(DatasetFormat::Parquet);

    // Quantization
    let quant = QuantizationConfig::default()
        .quantization_type(QuantizationType::Int8);

    // Export
    let export = ExportConfig::default()
        .format(ExportFormat::Apr);
}
```

## Model Stages

- `Development` - In development
- `Staging` - Ready for testing
- `Production` - Production deployment
- `Archived` - Archived/retired

## Dataset Formats

- `Parquet` - Columnar format (default)
- `Csv` - CSV files
- `JsonLines` - JSON Lines format
- `Arrow` - Apache Arrow IPC
- `Text` - Plain text files

## Quantization Types

- `Int8` - 8-bit integer (8x compression)
- `Int4` - 4-bit integer (16x compression)
- `Float16` - 16-bit float (4x compression)
- `Dynamic` - Dynamic quantization

## Export Formats

- `Apr` - Sovereign AI Stack native format
- `Json` - JSON format
- `Binary` - Binary format
- `OnnxMeta` - ONNX metadata (no runtime)
