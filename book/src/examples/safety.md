# Safety & Guardrails Example

Run with: `cargo run --example safety_demo`

## Overview

Demonstrates safety and privacy utilities:
- Content filtering
- Guardrails
- PII detection
- Anonymization
- Watermarking

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::safety::{
    ContentFilterConfig, ContentCategory, GuardrailType, GuardrailAction,
    PrivacyConfig, PiiType, AnonymizationMethod,
    WatermarkConfig, WatermarkType,
};

fn main() {
    // Content filtering
    let filter = ContentFilterConfig::default();
    println!("threshold: {}", filter.threshold);
    println!("strict_mode: {}", filter.strict_mode);

    // Guardrail types
    let guardrails = [
        GuardrailType::ContentFilter,
        GuardrailType::InputValidation,
        GuardrailType::RateLimiting,
        GuardrailType::OutputSanitization,
    ];

    // Privacy configuration
    let privacy = PrivacyConfig::default();
    println!("threshold: {}", privacy.threshold);

    // Watermarking
    let watermark = WatermarkConfig::default();
    println!("strength: {}", watermark.strength);
}
```

## Guardrail Types

- `ContentFilter` - Filter harmful content
- `InputValidation` - Validate inputs
- `RateLimiting` - Rate limit requests
- `OutputSanitization` - Sanitize outputs
- `TopicRestriction` - Restrict topics
- `FormatEnforcement` - Enforce output format

## Content Categories

- `Safe` - Safe content
- `Harmful` - Generally harmful
- `Violent` - Violence-related
- `Sexual` - Sexual content
- `Hateful` - Hate speech
- `SelfHarm` - Self-harm content
- `Illegal` - Illegal activities
- `Misinformation` - False information

## PII Types

- Email, Phone, SSN, Credit Card
- Name, Address, Date of Birth
- IP Address, License Plate, Medical ID

## Watermark Types

- `Soft` - Soft watermarking
- `Hard` - Hard watermarking
- `Semantic` - Semantic watermarking
- `Statistical` - Statistical watermarking
- `Neural` - Neural watermarking
- `Cryptographic` - Cryptographic watermarking
