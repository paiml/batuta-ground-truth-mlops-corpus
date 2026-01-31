//! Sampling Strategies for Text Generation
//!
//! Provides configuration for various text generation sampling strategies.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.generation import SamplingConfig, SamplingStrategy
//! config = create_sampling_config(temperature=0.7, top_p=0.9)
//! ```

/// Sampling strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplingStrategy {
    /// Greedy decoding (always pick highest prob)
    Greedy,
    /// Temperature-based sampling (default)
    #[default]
    Temperature,
    /// Top-k sampling
    TopK,
    /// Top-p (nucleus) sampling
    TopP,
    /// Combined top-k and top-p
    TopKTopP,
    /// Beam search
    BeamSearch,
    /// Contrastive search
    Contrastive,
}

impl SamplingStrategy {
    /// Get strategy name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Temperature => "temperature",
            Self::TopK => "top_k",
            Self::TopP => "top_p",
            Self::TopKTopP => "top_k_top_p",
            Self::BeamSearch => "beam_search",
            Self::Contrastive => "contrastive",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "greedy" => Some(Self::Greedy),
            "temperature" | "temp" => Some(Self::Temperature),
            "top_k" | "topk" => Some(Self::TopK),
            "top_p" | "topp" | "nucleus" => Some(Self::TopP),
            "top_k_top_p" | "combined" => Some(Self::TopKTopP),
            "beam_search" | "beam" => Some(Self::BeamSearch),
            "contrastive" => Some(Self::Contrastive),
            _ => None,
        }
    }

    /// List all strategies
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Greedy,
            Self::Temperature,
            Self::TopK,
            Self::TopP,
            Self::TopKTopP,
            Self::BeamSearch,
            Self::Contrastive,
        ]
    }

    /// Check if strategy is deterministic
    pub fn is_deterministic(&self) -> bool {
        matches!(self, Self::Greedy | Self::BeamSearch)
    }
}

/// Stopping criteria for generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StoppingCriteria {
    /// Stop at max length
    #[default]
    MaxLength,
    /// Stop at EOS token
    EosToken,
    /// Stop at max time
    MaxTime,
    /// Stop at custom condition
    Custom,
}

impl StoppingCriteria {
    /// Get criteria name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MaxLength => "max_length",
            Self::EosToken => "eos_token",
            Self::MaxTime => "max_time",
            Self::Custom => "custom",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "max_length" | "length" => Some(Self::MaxLength),
            "eos_token" | "eos" => Some(Self::EosToken),
            "max_time" | "time" => Some(Self::MaxTime),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    /// List all criteria
    pub fn list_all() -> Vec<Self> {
        vec![Self::MaxLength, Self::EosToken, Self::MaxTime, Self::Custom]
    }
}

/// Configuration for sampling-based generation
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Temperature for randomness (0.0 = deterministic, 1.0+ = random)
    pub temperature: f64,
    /// Top-k tokens to consider
    pub top_k: usize,
    /// Top-p (nucleus) probability mass
    pub top_p: f64,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,
    /// Minimum new tokens
    pub min_new_tokens: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Stopping criteria
    pub stopping_criteria: StoppingCriteria,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Temperature,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            max_new_tokens: 256,
            min_new_tokens: 0,
            seed: None,
            stopping_criteria: StoppingCriteria::MaxLength,
        }
    }
}

impl SamplingConfig {
    /// Create new sampling config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sampling strategy
    pub fn strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    /// Set repetition penalty
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set max new tokens
    pub fn max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// Set min new tokens
    pub fn min_new_tokens(mut self, n: usize) -> Self {
        self.min_new_tokens = n;
        self
    }

    /// Set seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set stopping criteria
    pub fn stopping_criteria(mut self, criteria: StoppingCriteria) -> Self {
        self.stopping_criteria = criteria;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("Temperature must be >= 0".to_string());
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("Top-p must be in (0, 1]".to_string());
        }
        if self.repetition_penalty < 1.0 {
            return Err("Repetition penalty must be >= 1.0".to_string());
        }
        if self.min_new_tokens > self.max_new_tokens {
            return Err("Min tokens cannot exceed max tokens".to_string());
        }
        Ok(())
    }

    /// Get effective vocabulary size after filtering
    pub fn effective_vocab_size(&self, vocab_size: usize) -> usize {
        match self.strategy {
            SamplingStrategy::Greedy => 1,
            SamplingStrategy::TopK => self.top_k.min(vocab_size),
            SamplingStrategy::TopP => (vocab_size as f64 * self.top_p) as usize,
            SamplingStrategy::TopKTopP => {
                let k_filtered = self.top_k.min(vocab_size);
                (k_filtered as f64 * self.top_p) as usize
            }
            _ => vocab_size,
        }
    }
}

/// Beam search configuration
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams
    pub num_beams: usize,
    /// Early stopping
    pub early_stopping: bool,
    /// Length penalty (> 1 favors longer, < 1 favors shorter)
    pub length_penalty: f64,
    /// Number of beam groups for diverse beam search
    pub num_beam_groups: usize,
    /// Diversity penalty
    pub diversity_penalty: f64,
    /// Number of sequences to return per batch
    pub num_return_sequences: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            early_stopping: true,
            length_penalty: 1.0,
            num_beam_groups: 1,
            diversity_penalty: 0.0,
            num_return_sequences: 1,
        }
    }
}

impl BeamSearchConfig {
    /// Create new beam search config
    pub fn new(num_beams: usize) -> Self {
        Self {
            num_beams,
            ..Default::default()
        }
    }

    /// Set early stopping
    pub fn early_stopping(mut self, enabled: bool) -> Self {
        self.early_stopping = enabled;
        self
    }

    /// Set length penalty
    pub fn length_penalty(mut self, penalty: f64) -> Self {
        self.length_penalty = penalty;
        self
    }

    /// Set number of beam groups
    pub fn num_beam_groups(mut self, groups: usize) -> Self {
        self.num_beam_groups = groups;
        self
    }

    /// Set diversity penalty
    pub fn diversity_penalty(mut self, penalty: f64) -> Self {
        self.diversity_penalty = penalty;
        self
    }

    /// Set number of return sequences
    pub fn num_return_sequences(mut self, n: usize) -> Self {
        self.num_return_sequences = n;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_beams == 0 {
            return Err("Number of beams must be > 0".to_string());
        }
        if self.num_beam_groups > self.num_beams {
            return Err("Beam groups cannot exceed num_beams".to_string());
        }
        if self.num_return_sequences > self.num_beams {
            return Err("Return sequences cannot exceed num_beams".to_string());
        }
        Ok(())
    }

    /// Check if using diverse beam search
    pub fn is_diverse(&self) -> bool {
        self.num_beam_groups > 1 || self.diversity_penalty > 0.0
    }
}

/// Contrastive search configuration
#[derive(Debug, Clone)]
pub struct ContrastiveConfig {
    /// Penalty alpha (0 = greedy, 1 = full contrastive)
    pub penalty_alpha: f64,
    /// Top-k for candidate selection
    pub top_k: usize,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            penalty_alpha: 0.6,
            top_k: 4,
        }
    }
}

impl ContrastiveConfig {
    /// Create new contrastive config
    pub fn new(penalty_alpha: f64) -> Self {
        Self {
            penalty_alpha,
            ..Default::default()
        }
    }

    /// Set top-k
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.penalty_alpha) {
            return Err("Penalty alpha must be in [0, 1]".to_string());
        }
        if self.top_k == 0 {
            return Err("Top-k must be > 0".to_string());
        }
        Ok(())
    }
}

/// Estimate generation memory usage
pub fn estimate_generation_memory(
    batch_size: usize,
    seq_length: usize,
    hidden_size: usize,
    num_layers: usize,
    num_beams: usize,
) -> u64 {
    // KV cache: 2 (k+v) * layers * batch * beams * seq * hidden * 2 (fp16)
    let kv_cache = 2 * num_layers * batch_size * num_beams * seq_length * hidden_size * 2;
    kv_cache as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_strategy_default() {
        assert_eq!(SamplingStrategy::default(), SamplingStrategy::Temperature);
    }

    #[test]
    fn test_sampling_strategy_as_str() {
        assert_eq!(SamplingStrategy::Greedy.as_str(), "greedy");
        assert_eq!(SamplingStrategy::TopK.as_str(), "top_k");
        assert_eq!(SamplingStrategy::TopP.as_str(), "top_p");
        assert_eq!(SamplingStrategy::BeamSearch.as_str(), "beam_search");
    }

    #[test]
    fn test_sampling_strategy_from_str() {
        assert_eq!(SamplingStrategy::parse("greedy"), Some(SamplingStrategy::Greedy));
        assert_eq!(SamplingStrategy::parse("nucleus"), Some(SamplingStrategy::TopP));
        assert_eq!(SamplingStrategy::parse("beam"), Some(SamplingStrategy::BeamSearch));
        assert_eq!(SamplingStrategy::parse("unknown"), None);
    }

    #[test]
    fn test_sampling_strategy_list_all() {
        assert_eq!(SamplingStrategy::list_all().len(), 7);
    }

    #[test]
    fn test_sampling_strategy_is_deterministic() {
        assert!(SamplingStrategy::Greedy.is_deterministic());
        assert!(SamplingStrategy::BeamSearch.is_deterministic());
        assert!(!SamplingStrategy::Temperature.is_deterministic());
        assert!(!SamplingStrategy::TopP.is_deterministic());
    }

    #[test]
    fn test_stopping_criteria_default() {
        assert_eq!(StoppingCriteria::default(), StoppingCriteria::MaxLength);
    }

    #[test]
    fn test_stopping_criteria_as_str() {
        assert_eq!(StoppingCriteria::MaxLength.as_str(), "max_length");
        assert_eq!(StoppingCriteria::EosToken.as_str(), "eos_token");
    }

    #[test]
    fn test_stopping_criteria_from_str() {
        assert_eq!(StoppingCriteria::parse("length"), Some(StoppingCriteria::MaxLength));
        assert_eq!(StoppingCriteria::parse("eos"), Some(StoppingCriteria::EosToken));
    }

    #[test]
    fn test_stopping_criteria_list_all() {
        assert_eq!(StoppingCriteria::list_all().len(), 4);
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.max_new_tokens, 256);
    }

    #[test]
    fn test_sampling_config_builder() {
        let config = SamplingConfig::new()
            .strategy(SamplingStrategy::TopP)
            .temperature(0.7)
            .top_k(40)
            .top_p(0.95)
            .repetition_penalty(1.2)
            .max_new_tokens(512)
            .min_new_tokens(10)
            .seed(42)
            .stopping_criteria(StoppingCriteria::EosToken);

        assert_eq!(config.strategy, SamplingStrategy::TopP);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.repetition_penalty, 1.2);
        assert_eq!(config.max_new_tokens, 512);
        assert_eq!(config.min_new_tokens, 10);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_sampling_config_validate() {
        let valid = SamplingConfig::default();
        assert!(valid.validate().is_ok());

        let neg_temp = SamplingConfig::new().temperature(-0.1);
        assert!(neg_temp.validate().is_err());

        let bad_top_p = SamplingConfig::new().top_p(1.5);
        assert!(bad_top_p.validate().is_err());

        let bad_rep = SamplingConfig::new().repetition_penalty(0.5);
        assert!(bad_rep.validate().is_err());

        let bad_tokens = SamplingConfig::new().min_new_tokens(100).max_new_tokens(50);
        assert!(bad_tokens.validate().is_err());
    }

    #[test]
    fn test_sampling_config_effective_vocab_size() {
        let greedy = SamplingConfig::new().strategy(SamplingStrategy::Greedy);
        assert_eq!(greedy.effective_vocab_size(50000), 1);

        let top_k = SamplingConfig::new().strategy(SamplingStrategy::TopK).top_k(100);
        assert_eq!(top_k.effective_vocab_size(50000), 100);

        let top_p = SamplingConfig::new().strategy(SamplingStrategy::TopP).top_p(0.5);
        assert_eq!(top_p.effective_vocab_size(50000), 25000);
    }

    #[test]
    fn test_beam_search_config_default() {
        let config = BeamSearchConfig::default();
        assert_eq!(config.num_beams, 4);
        assert!(config.early_stopping);
        assert_eq!(config.length_penalty, 1.0);
    }

    #[test]
    fn test_beam_search_config_builder() {
        let config = BeamSearchConfig::new(8)
            .early_stopping(false)
            .length_penalty(1.5)
            .num_beam_groups(2)
            .diversity_penalty(0.5)
            .num_return_sequences(4);

        assert_eq!(config.num_beams, 8);
        assert!(!config.early_stopping);
        assert_eq!(config.length_penalty, 1.5);
        assert_eq!(config.num_beam_groups, 2);
        assert_eq!(config.diversity_penalty, 0.5);
        assert_eq!(config.num_return_sequences, 4);
    }

    #[test]
    fn test_beam_search_config_validate() {
        let valid = BeamSearchConfig::default();
        assert!(valid.validate().is_ok());

        let zero_beams = BeamSearchConfig::new(0);
        assert!(zero_beams.validate().is_err());

        let too_many_groups = BeamSearchConfig::new(4).num_beam_groups(8);
        assert!(too_many_groups.validate().is_err());

        let too_many_returns = BeamSearchConfig::new(4).num_return_sequences(8);
        assert!(too_many_returns.validate().is_err());
    }

    #[test]
    fn test_beam_search_config_is_diverse() {
        let normal = BeamSearchConfig::default();
        assert!(!normal.is_diverse());

        let diverse_groups = BeamSearchConfig::new(8).num_beam_groups(2);
        assert!(diverse_groups.is_diverse());

        let diverse_penalty = BeamSearchConfig::new(4).diversity_penalty(0.5);
        assert!(diverse_penalty.is_diverse());
    }

    #[test]
    fn test_contrastive_config_default() {
        let config = ContrastiveConfig::default();
        assert_eq!(config.penalty_alpha, 0.6);
        assert_eq!(config.top_k, 4);
    }

    #[test]
    fn test_contrastive_config_builder() {
        let config = ContrastiveConfig::new(0.8).top_k(6);
        assert_eq!(config.penalty_alpha, 0.8);
        assert_eq!(config.top_k, 6);
    }

    #[test]
    fn test_contrastive_config_validate() {
        let valid = ContrastiveConfig::default();
        assert!(valid.validate().is_ok());

        let bad_alpha = ContrastiveConfig::new(1.5);
        assert!(bad_alpha.validate().is_err());

        let zero_k = ContrastiveConfig::new(0.5).top_k(0);
        assert!(zero_k.validate().is_err());
    }

    #[test]
    fn test_estimate_generation_memory() {
        let mem = estimate_generation_memory(1, 2048, 4096, 32, 1);
        assert!(mem > 0);
        
        // With beam search, memory scales with beams
        let mem_beams = estimate_generation_memory(1, 2048, 4096, 32, 4);
        assert_eq!(mem_beams, mem * 4);
    }
}
