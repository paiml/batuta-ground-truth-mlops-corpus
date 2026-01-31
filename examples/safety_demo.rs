//! Safety and guardrails demo

use batuta_ground_truth_mlops_corpus::safety::{
    ContentFilterConfig, ContentCategory, GuardrailType, GuardrailAction,
    PrivacyConfig, PiiType, AnonymizationMethod,
    WatermarkConfig, WatermarkType,
};

fn main() {
    println!("=== Safety & Guardrails Demo ===\n");

    // Content filtering
    println!("Content Filter Configuration:");
    let filter = ContentFilterConfig::default();

    println!("  threshold: {}", filter.threshold);
    println!("  strict_mode: {}", filter.strict_mode);

    // Guardrail types
    println!("\nGuardrail Types:");
    let types = [
        GuardrailType::ContentFilter,
        GuardrailType::InputValidation,
        GuardrailType::RateLimiting,
        GuardrailType::OutputSanitization,
    ];
    for t in &types {
        println!("  - {:?}", t);
    }

    // Content categories
    println!("\nContent Categories:");
    let categories = [
        ContentCategory::Harmful,
        ContentCategory::Violent,
        ContentCategory::Sexual,
        ContentCategory::Hateful,
        ContentCategory::SelfHarm,
    ];
    for cat in &categories {
        println!("  - {:?}", cat);
    }

    // Guardrail actions
    println!("\nGuardrail Actions:");
    let actions = [
        GuardrailAction::Block,
        GuardrailAction::Warn,
        GuardrailAction::Flag,
        GuardrailAction::Redact,
    ];
    for action in &actions {
        println!("  - {:?}", action);
    }

    // Privacy configuration
    println!("\n--- Privacy Configuration ---");
    let privacy = PrivacyConfig::default();

    println!("  threshold: {}", privacy.threshold);

    // PII types
    println!("\nPII Types:");
    let pii_types = [
        PiiType::Email,
        PiiType::Phone,
        PiiType::Ssn,
        PiiType::CreditCard,
        PiiType::Address,
        PiiType::Name,
        PiiType::DateOfBirth,
        PiiType::IpAddress,
    ];
    for pii in &pii_types {
        println!("  - {:?}", pii);
    }

    // Anonymization methods
    println!("\nAnonymization Methods:");
    let methods = [
        AnonymizationMethod::Mask,
        AnonymizationMethod::Remove,
        AnonymizationMethod::Hash,
        AnonymizationMethod::Replace,
    ];
    for method in &methods {
        println!("  - {:?}", method);
    }

    // Watermarking configuration
    println!("\n--- Watermarking Configuration ---");
    let watermark = WatermarkConfig::default();

    println!("  watermark_type: {:?}", watermark.watermark_type);
    println!("  strength: {}", watermark.strength);
    println!("  vocab_partition: {}", watermark.vocab_partition);

    // Watermark types
    println!("\nWatermark Types:");
    let wm_types = [
        WatermarkType::Soft,
        WatermarkType::Hard,
        WatermarkType::Semantic,
        WatermarkType::Statistical,
        WatermarkType::Neural,
        WatermarkType::Cryptographic,
    ];
    for wt in &wm_types {
        println!("  - {:?}", wt);
    }
}
