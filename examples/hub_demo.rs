//! Model hub and deployment demo

use batuta_ground_truth_mlops_corpus::hub::{
    RegistryConfig, ModelStage, VersioningScheme,
    parse_version,
    DatasetConfig, DatasetFormat,
};
use batuta_ground_truth_mlops_corpus::deployment::{
    QuantizationConfig, QuantizationType,
    ExportConfig, ExportFormat,
};

fn main() {
    println!("=== Model Hub & Deployment Demo ===\n");

    // Registry configuration
    println!("Model Registry Configuration:");
    let registry = RegistryConfig::new("sentiment-classifier")
        .namespace("paiml")
        .versioning_scheme(VersioningScheme::Semantic);

    println!("  name: {}", registry.name);
    println!("  namespace: {}", registry.namespace);
    println!("  versioning_scheme: {:?}", registry.versioning_scheme);

    // Model stages
    println!("\nModel Lifecycle Stages:");
    let stages = [
        ModelStage::Development,
        ModelStage::Staging,
        ModelStage::Production,
        ModelStage::Archived,
    ];
    for stage in &stages {
        println!("  - {:?}", stage);
    }

    // Version parsing
    println!("\n--- Version Parsing ---");
    let versions = ["1.0.0", "2.3.4", "0.1.0-beta"];
    for v in &versions {
        match parse_version(v) {
            Some(ver) => println!("  {} -> major:{}, minor:{}, patch:{}", v, ver.major, ver.minor, ver.patch),
            None => println!("  {} -> Invalid version", v),
        }
    }

    // Dataset configuration
    println!("\n--- Dataset Configuration ---");
    let dataset = DatasetConfig::new("imdb-reviews")
        .format(DatasetFormat::Parquet);

    println!("  name: {}", dataset.name);
    println!("  format: {:?}", dataset.format);

    // Dataset formats
    println!("\nDataset Formats:");
    let formats = [
        DatasetFormat::Csv,
        DatasetFormat::JsonLines,
        DatasetFormat::Parquet,
        DatasetFormat::Arrow,
    ];
    for fmt in &formats {
        println!("  - {:?}", fmt);
    }

    // Quantization
    println!("\n--- Quantization ---");
    let quant_int8 = QuantizationConfig::default()
        .quantization_type(QuantizationType::Int8);
    let quant_int4 = QuantizationConfig::default()
        .quantization_type(QuantizationType::Int4);

    println!("INT8 Quantization:");
    println!("  type: {:?}", quant_int8.quantization_type);

    println!("\nINT4 Quantization:");
    println!("  type: {:?}", quant_int4.quantization_type);

    // Quantization types
    println!("\nQuantization Types:");
    let quant_types = [
        QuantizationType::Int8,
        QuantizationType::Int4,
        QuantizationType::Float16,
        QuantizationType::Dynamic,
    ];
    for qt in &quant_types {
        println!("  - {:?}", qt);
    }

    // Export configuration
    println!("\n--- Export Configuration ---");
    let apr_export = ExportConfig::default()
        .format(ExportFormat::Apr);

    println!("APR Export (native format):");
    println!("  format: {:?}", apr_export.format);

    // Export formats
    println!("\nExport Formats:");
    let export_formats = [
        ExportFormat::Apr,
        ExportFormat::Json,
        ExportFormat::Binary,
        ExportFormat::OnnxMeta,
    ];
    for fmt in &export_formats {
        println!("  - {:?}", fmt);
    }
}
