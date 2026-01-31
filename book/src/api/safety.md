# Safety API

## Module: `batuta_ground_truth_mlops_corpus::safety`

### ContentFilterConfig

```rust
pub struct ContentFilterConfig {
    pub threshold: f32,
    pub strict_mode: bool,
}

impl ContentFilterConfig {
    pub fn new() -> Self;
    pub fn threshold(self, t: f32) -> Self;
    pub fn strict_mode(self, v: bool) -> Self;
}
```

### PrivacyConfig

```rust
pub struct PrivacyConfig {
    pub pii_types: HashSet<PiiType>,
    pub method: AnonymizationMethod,
    pub threshold: f32,
    pub placeholder: String,
}

impl PrivacyConfig {
    pub fn new() -> Self;
    pub fn pii_type(self, t: PiiType) -> Self;
    pub fn method(self, m: AnonymizationMethod) -> Self;
    pub fn threshold(self, t: f32) -> Self;
}
```

### WatermarkConfig

```rust
pub struct WatermarkConfig {
    pub watermark_type: WatermarkType,
    pub strength: f32,
    pub vocab_partition: f32,
}
```

### Enums

```rust
pub enum GuardrailType {
    ContentFilter,
    InputValidation,
    RateLimiting,
    OutputSanitization,
    TopicRestriction,
    FormatEnforcement,
}

pub enum ContentCategory {
    Safe, Harmful, Violent, Sexual, Hateful, SelfHarm, Illegal, Misinformation,
}

pub enum PiiType {
    Email, Phone, Ssn, CreditCard, Name, Address, DateOfBirth, IpAddress,
}

pub enum AnonymizationMethod {
    Replace, Mask, Hash, Generalize, Remove, Encrypt,
}
```
