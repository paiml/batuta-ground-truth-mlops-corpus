//! Privacy Protection
//!
//! PII detection and data anonymization for ML pipelines.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.safety import PiiType, PrivacyConfig, detect_pii
//! detected = detect_pii(text, types=[PiiType.EMAIL, PiiType.PHONE])
//! ```

use std::collections::HashSet;

/// PII (Personally Identifiable Information) type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PiiType {
    /// Email address (default)
    #[default]
    Email,
    /// Phone number
    Phone,
    /// Social security number
    Ssn,
    /// Credit card number
    CreditCard,
    /// Name
    Name,
    /// Address
    Address,
    /// Date of birth
    DateOfBirth,
    /// IP address
    IpAddress,
    /// License plate
    LicensePlate,
    /// Medical record
    MedicalId,
}

impl PiiType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Phone => "phone",
            Self::Ssn => "ssn",
            Self::CreditCard => "credit_card",
            Self::Name => "name",
            Self::Address => "address",
            Self::DateOfBirth => "dob",
            Self::IpAddress => "ip_address",
            Self::LicensePlate => "license_plate",
            Self::MedicalId => "medical_id",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "email" => Some(Self::Email),
            "phone" | "telephone" => Some(Self::Phone),
            "ssn" | "social_security" => Some(Self::Ssn),
            "credit_card" | "cc" => Some(Self::CreditCard),
            "name" | "person" => Some(Self::Name),
            "address" | "location" => Some(Self::Address),
            "dob" | "date_of_birth" | "birthday" => Some(Self::DateOfBirth),
            "ip_address" | "ip" => Some(Self::IpAddress),
            "license_plate" | "plate" => Some(Self::LicensePlate),
            "medical_id" | "medical" => Some(Self::MedicalId),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Email,
            Self::Phone,
            Self::Ssn,
            Self::CreditCard,
            Self::Name,
            Self::Address,
            Self::DateOfBirth,
            Self::IpAddress,
            Self::LicensePlate,
            Self::MedicalId,
        ]
    }

    /// Get sensitivity level (1-10)
    pub fn sensitivity(&self) -> u8 {
        match self {
            Self::Name => 3,
            Self::Email => 4,
            Self::Phone => 5,
            Self::Address => 6,
            Self::DateOfBirth => 6,
            Self::IpAddress => 4,
            Self::LicensePlate => 5,
            Self::CreditCard => 9,
            Self::Ssn => 10,
            Self::MedicalId => 10,
        }
    }

    /// Check if highly sensitive
    pub fn is_highly_sensitive(&self) -> bool {
        self.sensitivity() >= 8
    }
}

/// Anonymization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnonymizationMethod {
    /// Replace with placeholder (default)
    #[default]
    Replace,
    /// Mask characters
    Mask,
    /// Hash value
    Hash,
    /// Generalize (e.g., age range)
    Generalize,
    /// Remove entirely
    Remove,
    /// Encrypt
    Encrypt,
}

impl AnonymizationMethod {
    /// Get method name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Replace => "replace",
            Self::Mask => "mask",
            Self::Hash => "hash",
            Self::Generalize => "generalize",
            Self::Remove => "remove",
            Self::Encrypt => "encrypt",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "replace" | "placeholder" => Some(Self::Replace),
            "mask" | "redact" => Some(Self::Mask),
            "hash" | "digest" => Some(Self::Hash),
            "generalize" | "aggregate" => Some(Self::Generalize),
            "remove" | "delete" => Some(Self::Remove),
            "encrypt" | "cipher" => Some(Self::Encrypt),
            _ => None,
        }
    }

    /// Check if method is reversible
    pub fn is_reversible(&self) -> bool {
        matches!(self, Self::Encrypt)
    }
}

/// Privacy compliance standard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComplianceStandard {
    /// GDPR (default)
    #[default]
    Gdpr,
    /// CCPA (California)
    Ccpa,
    /// HIPAA (Healthcare)
    Hipaa,
    /// PCI-DSS (Payment)
    PciDss,
    /// SOC 2
    Soc2,
    /// Custom policy
    Custom,
}

impl ComplianceStandard {
    /// Get standard name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gdpr => "gdpr",
            Self::Ccpa => "ccpa",
            Self::Hipaa => "hipaa",
            Self::PciDss => "pci_dss",
            Self::Soc2 => "soc2",
            Self::Custom => "custom",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gdpr" => Some(Self::Gdpr),
            "ccpa" | "california" => Some(Self::Ccpa),
            "hipaa" | "healthcare" => Some(Self::Hipaa),
            "pci_dss" | "pci" | "payment" => Some(Self::PciDss),
            "soc2" | "soc" => Some(Self::Soc2),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    /// Get required PII types to protect
    pub fn required_pii_types(&self) -> Vec<PiiType> {
        match self {
            Self::Gdpr | Self::Ccpa => PiiType::list_all(),
            Self::Hipaa => vec![
                PiiType::Name,
                PiiType::DateOfBirth,
                PiiType::Ssn,
                PiiType::MedicalId,
                PiiType::Address,
                PiiType::Phone,
                PiiType::Email,
            ],
            Self::PciDss => vec![PiiType::CreditCard, PiiType::Name, PiiType::Address],
            Self::Soc2 => vec![
                PiiType::Email,
                PiiType::Phone,
                PiiType::Name,
                PiiType::Address,
            ],
            Self::Custom => Vec::new(),
        }
    }
}

/// PII detection result
#[derive(Debug, Clone)]
pub struct PiiDetection {
    /// Detected PII type
    pub pii_type: PiiType,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Matched text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
}

impl PiiDetection {
    /// Create new detection
    pub fn new(pii_type: PiiType, start: usize, end: usize, text: impl Into<String>) -> Self {
        Self {
            pii_type,
            start,
            end,
            text: text.into(),
            confidence: 1.0,
        }
    }

    /// Set confidence
    pub fn confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Get text length
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Privacy configuration
#[derive(Debug, Clone)]
pub struct PrivacyConfig {
    /// PII types to detect
    pub pii_types: HashSet<PiiType>,
    /// Anonymization method
    pub method: AnonymizationMethod,
    /// Compliance standard
    pub standard: ComplianceStandard,
    /// Detection threshold
    pub threshold: f32,
    /// Replacement placeholder
    pub placeholder: String,
    /// Mask character
    pub mask_char: char,
    /// Enable logging of detections
    pub log_detections: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        let mut pii_types = HashSet::new();
        pii_types.insert(PiiType::Email);
        pii_types.insert(PiiType::Phone);
        pii_types.insert(PiiType::Ssn);
        pii_types.insert(PiiType::CreditCard);

        Self {
            pii_types,
            method: AnonymizationMethod::Replace,
            standard: ComplianceStandard::Gdpr,
            threshold: 0.8,
            placeholder: "[REDACTED]".to_string(),
            mask_char: '*',
            log_detections: false,
        }
    }
}

impl PrivacyConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set PII types
    pub fn pii_types(mut self, types: HashSet<PiiType>) -> Self {
        self.pii_types = types;
        self
    }

    /// Add PII type
    pub fn pii_type(mut self, t: PiiType) -> Self {
        self.pii_types.insert(t);
        self
    }

    /// Set method
    pub fn method(mut self, m: AnonymizationMethod) -> Self {
        self.method = m;
        self
    }

    /// Set compliance standard
    pub fn standard(mut self, s: ComplianceStandard) -> Self {
        self.standard = s;
        self.pii_types.extend(s.required_pii_types());
        self
    }

    /// Set threshold
    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Set placeholder
    pub fn placeholder(mut self, p: impl Into<String>) -> Self {
        self.placeholder = p.into();
        self
    }

    /// Set mask character
    pub fn mask_char(mut self, c: char) -> Self {
        self.mask_char = c;
        self
    }

    /// Enable logging
    pub fn log_detections(mut self, enabled: bool) -> Self {
        self.log_detections = enabled;
        self
    }

    /// Check if type is monitored
    pub fn monitors_type(&self, t: PiiType) -> bool {
        self.pii_types.contains(&t)
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.pii_types.is_empty() {
            return Err("Must specify at least one PII type".to_string());
        }
        if self.placeholder.is_empty() && self.method == AnonymizationMethod::Replace {
            return Err("Placeholder required for replace method".to_string());
        }
        Ok(())
    }
}

/// Anonymization result
#[derive(Debug, Clone)]
pub struct AnonymizationResult {
    /// Anonymized text
    pub text: String,
    /// Detections found
    pub detections: Vec<PiiDetection>,
    /// Method used
    pub method: AnonymizationMethod,
    /// Total PII instances found
    pub pii_count: usize,
}

impl AnonymizationResult {
    /// Create new result
    pub fn new(text: String, method: AnonymizationMethod) -> Self {
        Self {
            text,
            detections: Vec::new(),
            method,
            pii_count: 0,
        }
    }

    /// Add detection
    pub fn with_detection(mut self, detection: PiiDetection) -> Self {
        self.detections.push(detection);
        self.pii_count += 1;
        self
    }

    /// Check if PII was found
    pub fn has_pii(&self) -> bool {
        self.pii_count > 0
    }

    /// Get unique PII types found
    pub fn unique_types(&self) -> HashSet<PiiType> {
        self.detections.iter().map(|d| d.pii_type).collect()
    }
}

/// Privacy statistics
#[derive(Debug, Clone, Default)]
pub struct PrivacyStats {
    /// Total texts processed
    pub texts_processed: usize,
    /// Total PII detected
    pub total_pii: usize,
    /// PII by type
    pub pii_by_type: std::collections::HashMap<String, usize>,
    /// Texts with PII
    pub texts_with_pii: usize,
    /// Total characters anonymized
    pub chars_anonymized: usize,
}

impl PrivacyStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record result
    pub fn record(&mut self, result: &AnonymizationResult) {
        self.texts_processed += 1;
        self.total_pii += result.pii_count;
        if result.has_pii() {
            self.texts_with_pii += 1;
        }
        for detection in &result.detections {
            *self
                .pii_by_type
                .entry(detection.pii_type.as_str().to_string())
                .or_insert(0) += 1;
            self.chars_anonymized += detection.len();
        }
    }

    /// Get PII rate
    pub fn pii_rate(&self) -> f64 {
        if self.texts_processed == 0 {
            return 0.0;
        }
        self.texts_with_pii as f64 / self.texts_processed as f64
    }

    /// Get average PII per text
    pub fn avg_pii_per_text(&self) -> f64 {
        if self.texts_processed == 0 {
            return 0.0;
        }
        self.total_pii as f64 / self.texts_processed as f64
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Privacy: {} texts, {} PII ({:.1}% with PII)",
            self.texts_processed,
            self.total_pii,
            self.pii_rate() * 100.0
        )
    }
}

/// Data retention configuration
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    /// Retention period in days
    pub retention_days: u32,
    /// Auto-delete enabled
    pub auto_delete: bool,
    /// Require explicit consent
    pub require_consent: bool,
    /// Anonymize after retention
    pub anonymize_after: bool,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            retention_days: 90,
            auto_delete: false,
            require_consent: true,
            anonymize_after: true,
        }
    }
}

impl RetentionConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set retention days
    pub fn retention_days(mut self, days: u32) -> Self {
        self.retention_days = days;
        self
    }

    /// Enable auto delete
    pub fn auto_delete(mut self, enabled: bool) -> Self {
        self.auto_delete = enabled;
        self
    }

    /// Require consent
    pub fn require_consent(mut self, required: bool) -> Self {
        self.require_consent = required;
        self
    }

    /// Anonymize after retention
    pub fn anonymize_after(mut self, enabled: bool) -> Self {
        self.anonymize_after = enabled;
        self
    }

    /// Check if data should be retained
    pub fn should_retain(&self, age_days: u32) -> bool {
        age_days < self.retention_days
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.retention_days == 0 && !self.auto_delete {
            return Err("Zero retention requires auto-delete".to_string());
        }
        Ok(())
    }
}

/// Mask text with mask character
pub fn mask_text(text: &str, mask_char: char, visible_prefix: usize, visible_suffix: usize) -> String {
    let len = text.len();
    if len <= visible_prefix + visible_suffix {
        return mask_char.to_string().repeat(len);
    }

    let prefix: String = text.chars().take(visible_prefix).collect();
    let suffix: String = text.chars().skip(len - visible_suffix).collect();
    let mask_len = len - visible_prefix - visible_suffix;

    format!("{}{}{}", prefix, mask_char.to_string().repeat(mask_len), suffix)
}

/// Calculate privacy risk score (0.0-1.0)
pub fn calculate_privacy_risk(detections: &[PiiDetection]) -> f32 {
    if detections.is_empty() {
        return 0.0;
    }

    let total_sensitivity: f32 = detections
        .iter()
        .map(|d| d.pii_type.sensitivity() as f32 * d.confidence / 10.0)
        .sum();

    let max_risk = detections.len() as f32;
    (total_sensitivity / max_risk).clamp(0.0, 1.0)
}

/// Simple email pattern check (basic, not comprehensive)
pub fn looks_like_email(text: &str) -> bool {
    text.contains('@') && text.contains('.') && text.len() >= 5
}

/// Simple phone pattern check (basic, US format)
pub fn looks_like_phone(text: &str) -> bool {
    let digits: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.len() >= 10 && digits.len() <= 15
}

/// Simple credit card pattern check (Luhn-like length check)
pub fn looks_like_credit_card(text: &str) -> bool {
    let digits: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.len() >= 13 && digits.len() <= 19
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_type_default() {
        assert_eq!(PiiType::default(), PiiType::Email);
    }

    #[test]
    fn test_pii_type_as_str() {
        assert_eq!(PiiType::Email.as_str(), "email");
        assert_eq!(PiiType::Phone.as_str(), "phone");
        assert_eq!(PiiType::Ssn.as_str(), "ssn");
        assert_eq!(PiiType::CreditCard.as_str(), "credit_card");
        assert_eq!(PiiType::Name.as_str(), "name");
        assert_eq!(PiiType::Address.as_str(), "address");
        assert_eq!(PiiType::DateOfBirth.as_str(), "dob");
        assert_eq!(PiiType::IpAddress.as_str(), "ip_address");
        assert_eq!(PiiType::LicensePlate.as_str(), "license_plate");
        assert_eq!(PiiType::MedicalId.as_str(), "medical_id");
    }

    #[test]
    fn test_pii_type_parse() {
        assert_eq!(PiiType::parse("email"), Some(PiiType::Email));
        assert_eq!(PiiType::parse("telephone"), Some(PiiType::Phone));
        assert_eq!(PiiType::parse("social_security"), Some(PiiType::Ssn));
        assert_eq!(PiiType::parse("cc"), Some(PiiType::CreditCard));
        assert_eq!(PiiType::parse("person"), Some(PiiType::Name));
        assert_eq!(PiiType::parse("location"), Some(PiiType::Address));
        assert_eq!(PiiType::parse("birthday"), Some(PiiType::DateOfBirth));
        assert_eq!(PiiType::parse("ip"), Some(PiiType::IpAddress));
        assert_eq!(PiiType::parse("plate"), Some(PiiType::LicensePlate));
        assert_eq!(PiiType::parse("medical"), Some(PiiType::MedicalId));
        assert_eq!(PiiType::parse("unknown"), None);
    }

    #[test]
    fn test_pii_type_list_all() {
        assert_eq!(PiiType::list_all().len(), 10);
    }

    #[test]
    fn test_pii_type_sensitivity() {
        assert_eq!(PiiType::Name.sensitivity(), 3);
        assert_eq!(PiiType::Ssn.sensitivity(), 10);
        assert_eq!(PiiType::CreditCard.sensitivity(), 9);
    }

    #[test]
    fn test_pii_type_is_highly_sensitive() {
        assert!(!PiiType::Name.is_highly_sensitive());
        assert!(!PiiType::Email.is_highly_sensitive());
        assert!(PiiType::Ssn.is_highly_sensitive());
        assert!(PiiType::CreditCard.is_highly_sensitive());
        assert!(PiiType::MedicalId.is_highly_sensitive());
    }

    #[test]
    fn test_anonymization_method_as_str() {
        assert_eq!(AnonymizationMethod::Replace.as_str(), "replace");
        assert_eq!(AnonymizationMethod::Mask.as_str(), "mask");
        assert_eq!(AnonymizationMethod::Hash.as_str(), "hash");
        assert_eq!(AnonymizationMethod::Generalize.as_str(), "generalize");
        assert_eq!(AnonymizationMethod::Remove.as_str(), "remove");
        assert_eq!(AnonymizationMethod::Encrypt.as_str(), "encrypt");
    }

    #[test]
    fn test_anonymization_method_parse() {
        assert_eq!(AnonymizationMethod::parse("placeholder"), Some(AnonymizationMethod::Replace));
        assert_eq!(AnonymizationMethod::parse("redact"), Some(AnonymizationMethod::Mask));
        assert_eq!(AnonymizationMethod::parse("digest"), Some(AnonymizationMethod::Hash));
        assert_eq!(AnonymizationMethod::parse("aggregate"), Some(AnonymizationMethod::Generalize));
        assert_eq!(AnonymizationMethod::parse("delete"), Some(AnonymizationMethod::Remove));
        assert_eq!(AnonymizationMethod::parse("cipher"), Some(AnonymizationMethod::Encrypt));
        assert_eq!(AnonymizationMethod::parse("unknown"), None);
    }

    #[test]
    fn test_anonymization_method_is_reversible() {
        assert!(!AnonymizationMethod::Replace.is_reversible());
        assert!(!AnonymizationMethod::Mask.is_reversible());
        assert!(!AnonymizationMethod::Hash.is_reversible());
        assert!(AnonymizationMethod::Encrypt.is_reversible());
    }

    #[test]
    fn test_compliance_standard_as_str() {
        assert_eq!(ComplianceStandard::Gdpr.as_str(), "gdpr");
        assert_eq!(ComplianceStandard::Ccpa.as_str(), "ccpa");
        assert_eq!(ComplianceStandard::Hipaa.as_str(), "hipaa");
        assert_eq!(ComplianceStandard::PciDss.as_str(), "pci_dss");
        assert_eq!(ComplianceStandard::Soc2.as_str(), "soc2");
        assert_eq!(ComplianceStandard::Custom.as_str(), "custom");
    }

    #[test]
    fn test_compliance_standard_parse() {
        assert_eq!(ComplianceStandard::parse("gdpr"), Some(ComplianceStandard::Gdpr));
        assert_eq!(ComplianceStandard::parse("california"), Some(ComplianceStandard::Ccpa));
        assert_eq!(ComplianceStandard::parse("healthcare"), Some(ComplianceStandard::Hipaa));
        assert_eq!(ComplianceStandard::parse("payment"), Some(ComplianceStandard::PciDss));
        assert_eq!(ComplianceStandard::parse("soc"), Some(ComplianceStandard::Soc2));
        assert_eq!(ComplianceStandard::parse("custom"), Some(ComplianceStandard::Custom));
        assert_eq!(ComplianceStandard::parse("unknown"), None);
    }

    #[test]
    fn test_compliance_standard_required_types() {
        let gdpr_types = ComplianceStandard::Gdpr.required_pii_types();
        assert_eq!(gdpr_types.len(), 10); // All types

        let pci_types = ComplianceStandard::PciDss.required_pii_types();
        assert!(pci_types.contains(&PiiType::CreditCard));

        let custom_types = ComplianceStandard::Custom.required_pii_types();
        assert!(custom_types.is_empty());
    }

    #[test]
    fn test_pii_detection_new() {
        let detection = PiiDetection::new(PiiType::Email, 0, 15, "test@example.com");
        assert_eq!(detection.pii_type, PiiType::Email);
        assert_eq!(detection.start, 0);
        assert_eq!(detection.end, 15);
        assert_eq!(detection.confidence, 1.0);
    }

    #[test]
    fn test_pii_detection_confidence() {
        let detection = PiiDetection::new(PiiType::Phone, 0, 10, "1234567890")
            .confidence(0.85);
        assert!((detection.confidence - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_pii_detection_len() {
        let detection = PiiDetection::new(PiiType::Email, 5, 20, "test@example.com");
        assert_eq!(detection.len(), 15);
        assert!(!detection.is_empty());
    }

    #[test]
    fn test_privacy_config_default() {
        let config = PrivacyConfig::default();
        assert!(config.pii_types.contains(&PiiType::Email));
        assert!(config.pii_types.contains(&PiiType::Phone));
        assert_eq!(config.method, AnonymizationMethod::Replace);
        assert_eq!(config.standard, ComplianceStandard::Gdpr);
    }

    #[test]
    fn test_privacy_config_builder() {
        let config = PrivacyConfig::new()
            .pii_type(PiiType::Name)
            .method(AnonymizationMethod::Mask)
            .standard(ComplianceStandard::Hipaa)
            .threshold(0.9)
            .placeholder("[PII]")
            .mask_char('#')
            .log_detections(true);

        assert!(config.monitors_type(PiiType::Name));
        assert_eq!(config.method, AnonymizationMethod::Mask);
        assert_eq!(config.threshold, 0.9);
        assert_eq!(config.placeholder, "[PII]");
        assert_eq!(config.mask_char, '#');
        assert!(config.log_detections);
    }

    #[test]
    fn test_privacy_config_validate() {
        let valid = PrivacyConfig::default();
        assert!(valid.validate().is_ok());

        let no_types = PrivacyConfig::new().pii_types(HashSet::new());
        assert!(no_types.validate().is_err());

        let empty_placeholder = PrivacyConfig::new()
            .method(AnonymizationMethod::Replace)
            .placeholder("");
        assert!(empty_placeholder.validate().is_err());
    }

    #[test]
    fn test_anonymization_result_new() {
        let result = AnonymizationResult::new("Hello".to_string(), AnonymizationMethod::Replace);
        assert_eq!(result.text, "Hello");
        assert!(!result.has_pii());
    }

    #[test]
    fn test_anonymization_result_with_detection() {
        let result = AnonymizationResult::new("Hello".to_string(), AnonymizationMethod::Replace)
            .with_detection(PiiDetection::new(PiiType::Email, 0, 5, "email"))
            .with_detection(PiiDetection::new(PiiType::Phone, 6, 10, "phone"));

        assert!(result.has_pii());
        assert_eq!(result.pii_count, 2);
        assert_eq!(result.unique_types().len(), 2);
    }

    #[test]
    fn test_privacy_stats_new() {
        let stats = PrivacyStats::new();
        assert_eq!(stats.texts_processed, 0);
        assert_eq!(stats.pii_rate(), 0.0);
    }

    #[test]
    fn test_privacy_stats_record() {
        let mut stats = PrivacyStats::new();

        let result1 = AnonymizationResult::new("clean".to_string(), AnonymizationMethod::Replace);
        let result2 = AnonymizationResult::new("pii".to_string(), AnonymizationMethod::Replace)
            .with_detection(PiiDetection::new(PiiType::Email, 0, 5, "email"));

        stats.record(&result1);
        stats.record(&result2);

        assert_eq!(stats.texts_processed, 2);
        assert_eq!(stats.texts_with_pii, 1);
        assert_eq!(stats.total_pii, 1);
        assert!((stats.pii_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_privacy_stats_avg_pii() {
        let mut stats = PrivacyStats::new();

        for _ in 0..4 {
            let result = AnonymizationResult::new("text".to_string(), AnonymizationMethod::Replace)
                .with_detection(PiiDetection::new(PiiType::Email, 0, 5, "e"))
                .with_detection(PiiDetection::new(PiiType::Phone, 0, 5, "p"));
            stats.record(&result);
        }

        assert!((stats.avg_pii_per_text() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_privacy_stats_format() {
        let mut stats = PrivacyStats::new();
        stats.record(&AnonymizationResult::new("t".to_string(), AnonymizationMethod::Replace));
        let formatted = stats.format();

        assert!(formatted.contains("1 texts"));
        assert!(formatted.contains("0 PII"));
    }

    #[test]
    fn test_retention_config_default() {
        let config = RetentionConfig::default();
        assert_eq!(config.retention_days, 90);
        assert!(!config.auto_delete);
        assert!(config.require_consent);
    }

    #[test]
    fn test_retention_config_builder() {
        let config = RetentionConfig::new()
            .retention_days(30)
            .auto_delete(true)
            .require_consent(false)
            .anonymize_after(false);

        assert_eq!(config.retention_days, 30);
        assert!(config.auto_delete);
        assert!(!config.require_consent);
        assert!(!config.anonymize_after);
    }

    #[test]
    fn test_retention_config_should_retain() {
        let config = RetentionConfig::new().retention_days(30);
        assert!(config.should_retain(29));
        assert!(!config.should_retain(30));
        assert!(!config.should_retain(31));
    }

    #[test]
    fn test_retention_config_validate() {
        let valid = RetentionConfig::default();
        assert!(valid.validate().is_ok());

        let zero_no_delete = RetentionConfig::new()
            .retention_days(0)
            .auto_delete(false);
        assert!(zero_no_delete.validate().is_err());

        let zero_with_delete = RetentionConfig::new()
            .retention_days(0)
            .auto_delete(true);
        assert!(zero_with_delete.validate().is_ok());
    }

    #[test]
    fn test_mask_text() {
        assert_eq!(mask_text("1234567890", '*', 2, 2), "12******90");
        assert_eq!(mask_text("abc", '*', 1, 1), "a*c");
        assert_eq!(mask_text("ab", '*', 1, 1), "**");
        assert_eq!(mask_text("a", '*', 1, 1), "*");
    }

    #[test]
    fn test_mask_text_different_char() {
        assert_eq!(mask_text("secret123", '#', 3, 2), "sec####23");
    }

    #[test]
    fn test_calculate_privacy_risk_empty() {
        assert_eq!(calculate_privacy_risk(&[]), 0.0);
    }

    #[test]
    fn test_calculate_privacy_risk_low() {
        let detections = vec![
            PiiDetection::new(PiiType::Name, 0, 5, "name").confidence(0.8),
        ];
        let risk = calculate_privacy_risk(&detections);
        assert!(risk < 0.5); // Name has low sensitivity
    }

    #[test]
    fn test_calculate_privacy_risk_high() {
        let detections = vec![
            PiiDetection::new(PiiType::Ssn, 0, 11, "123-45-6789").confidence(1.0),
        ];
        let risk = calculate_privacy_risk(&detections);
        assert!(risk > 0.8); // SSN has max sensitivity
    }

    #[test]
    fn test_looks_like_email() {
        assert!(looks_like_email("test@example.com"));
        assert!(looks_like_email("a@b.c"));
        assert!(!looks_like_email("not an email"));
        assert!(!looks_like_email("@."));
    }

    #[test]
    fn test_looks_like_phone() {
        assert!(looks_like_phone("1234567890"));
        assert!(looks_like_phone("(123) 456-7890"));
        assert!(looks_like_phone("+1-123-456-7890"));
        assert!(!looks_like_phone("12345"));
        assert!(!looks_like_phone("not a phone"));
    }

    #[test]
    fn test_looks_like_credit_card() {
        assert!(looks_like_credit_card("4111111111111111"));
        assert!(looks_like_credit_card("4111-1111-1111-1111"));
        assert!(!looks_like_credit_card("12345"));
        assert!(!looks_like_credit_card("not a card"));
    }
}
