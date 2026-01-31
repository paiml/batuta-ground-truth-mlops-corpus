//! Prompting Utilities
//!
//! Few-shot prompting, chain-of-thought, and prompt templates.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.generation import PromptTemplate, FewShotConfig
//! template = create_prompt_template(system="You are helpful", user="{query}")
//! ```

/// Prompt format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptFormat {
    /// Plain text format (default)
    #[default]
    Plain,
    /// ChatML format
    ChatML,
    /// Llama/Alpaca instruction format
    Alpaca,
    /// Vicuna format
    Vicuna,
    /// Zephyr format
    Zephyr,
}

impl PromptFormat {
    /// Get format name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::ChatML => "chatml",
            Self::Alpaca => "alpaca",
            Self::Vicuna => "vicuna",
            Self::Zephyr => "zephyr",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "plain" | "text" => Some(Self::Plain),
            "chatml" | "chat_ml" => Some(Self::ChatML),
            "alpaca" | "llama" => Some(Self::Alpaca),
            "vicuna" => Some(Self::Vicuna),
            "zephyr" => Some(Self::Zephyr),
            _ => None,
        }
    }

    /// List all formats
    pub fn list_all() -> Vec<Self> {
        vec![Self::Plain, Self::ChatML, Self::Alpaca, Self::Vicuna, Self::Zephyr]
    }

    /// Get system prompt prefix
    pub fn system_prefix(&self) -> &'static str {
        match self {
            Self::Plain => "",
            Self::ChatML => "<|im_start|>system\n",
            Self::Alpaca => "### Instruction:\n",
            Self::Vicuna => "SYSTEM: ",
            Self::Zephyr => "<|system|>\n",
        }
    }

    /// Get system prompt suffix
    pub fn system_suffix(&self) -> &'static str {
        match self {
            Self::Plain => "\n\n",
            Self::ChatML => "<|im_end|>\n",
            Self::Alpaca => "\n\n",
            Self::Vicuna => "\n\n",
            Self::Zephyr => "</s>\n",
        }
    }
}

/// Few-shot example selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FewShotStrategy {
    /// Random selection (default)
    #[default]
    Random,
    /// Similarity-based selection
    Similar,
    /// Diverse selection
    Diverse,
    /// Fixed/manual selection
    Fixed,
}

impl FewShotStrategy {
    /// Get strategy name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Similar => "similar",
            Self::Diverse => "diverse",
            Self::Fixed => "fixed",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "random" => Some(Self::Random),
            "similar" | "similarity" => Some(Self::Similar),
            "diverse" | "diversity" => Some(Self::Diverse),
            "fixed" | "manual" => Some(Self::Fixed),
            _ => None,
        }
    }

    /// List all strategies
    pub fn list_all() -> Vec<Self> {
        vec![Self::Random, Self::Similar, Self::Diverse, Self::Fixed]
    }
}

/// Reasoning type for chain-of-thought
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningType {
    /// No explicit reasoning
    #[default]
    None,
    /// Step-by-step reasoning
    StepByStep,
    /// Let's think step by step
    LetThink,
    /// Self-consistency with voting
    SelfConsistency,
    /// Tree of thoughts
    TreeOfThoughts,
}

impl ReasoningType {
    /// Get reasoning type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::StepByStep => "step_by_step",
            Self::LetThink => "let_think",
            Self::SelfConsistency => "self_consistency",
            Self::TreeOfThoughts => "tree_of_thoughts",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "step_by_step" | "cot" => Some(Self::StepByStep),
            "let_think" | "lets_think" => Some(Self::LetThink),
            "self_consistency" | "sc" => Some(Self::SelfConsistency),
            "tree_of_thoughts" | "tot" => Some(Self::TreeOfThoughts),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::None,
            Self::StepByStep,
            Self::LetThink,
            Self::SelfConsistency,
            Self::TreeOfThoughts,
        ]
    }

    /// Get the prompt suffix for this reasoning type
    pub fn prompt_suffix(&self) -> &'static str {
        match self {
            Self::None => "",
            Self::StepByStep => "\n\nLet me solve this step by step:\n",
            Self::LetThink => "\n\nLet's think step by step.",
            Self::SelfConsistency => "\n\nI'll reason through this multiple ways:",
            Self::TreeOfThoughts => "\n\nLet me explore different approaches:",
        }
    }
}

/// A few-shot example
#[derive(Debug, Clone)]
pub struct FewShotExample {
    /// Input/question
    pub input: String,
    /// Expected output/answer
    pub output: String,
    /// Optional reasoning/explanation
    pub reasoning: Option<String>,
}

impl FewShotExample {
    /// Create new example
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            reasoning: None,
        }
    }

    /// Add reasoning
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Format the example
    pub fn format(&self, input_prefix: &str, output_prefix: &str) -> String {
        let mut s = format!("{}{}\n", input_prefix, self.input);
        if let Some(ref reasoning) = self.reasoning {
            s.push_str(reasoning);
            s.push('\n');
        }
        s.push_str(&format!("{}{}", output_prefix, self.output));
        s
    }
}

/// Few-shot configuration
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    /// Examples to use
    pub examples: Vec<FewShotExample>,
    /// Selection strategy
    pub strategy: FewShotStrategy,
    /// Maximum number of examples
    pub max_examples: usize,
    /// Separator between examples
    pub separator: String,
    /// Input prefix
    pub input_prefix: String,
    /// Output prefix
    pub output_prefix: String,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            examples: Vec::new(),
            strategy: FewShotStrategy::Random,
            max_examples: 3,
            separator: String::from("\n\n"),
            input_prefix: String::from("Input: "),
            output_prefix: String::from("Output: "),
        }
    }
}

impl FewShotConfig {
    /// Create new few-shot config
    pub fn new() -> Self {
        Self::default()
    }

    /// Add example
    pub fn example(mut self, example: FewShotExample) -> Self {
        self.examples.push(example);
        self
    }

    /// Set strategy
    pub fn strategy(mut self, strategy: FewShotStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set max examples
    pub fn max_examples(mut self, n: usize) -> Self {
        self.max_examples = n;
        self
    }

    /// Set separator
    pub fn separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Set input prefix
    pub fn input_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.input_prefix = prefix.into();
        self
    }

    /// Set output prefix
    pub fn output_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.output_prefix = prefix.into();
        self
    }

    /// Format all examples
    pub fn format_examples(&self) -> String {
        let examples_to_use = self.examples.iter().take(self.max_examples);
        examples_to_use
            .map(|ex| ex.format(&self.input_prefix, &self.output_prefix))
            .collect::<Vec<_>>()
            .join(&self.separator)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_examples == 0 {
            return Err("Max examples must be > 0".to_string());
        }
        Ok(())
    }
}

/// Prompt template
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// System prompt
    pub system: String,
    /// User prompt template (with {placeholders})
    pub user_template: String,
    /// Prompt format
    pub format: PromptFormat,
    /// Few-shot config
    pub few_shot: Option<FewShotConfig>,
    /// Reasoning type
    pub reasoning: ReasoningType,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self {
            system: String::from("You are a helpful assistant."),
            user_template: String::from("{input}"),
            format: PromptFormat::Plain,
            few_shot: None,
            reasoning: ReasoningType::None,
        }
    }
}

impl PromptTemplate {
    /// Create new prompt template
    pub fn new() -> Self {
        Self::default()
    }

    /// Set system prompt
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into();
        self
    }

    /// Set user template
    pub fn user_template(mut self, template: impl Into<String>) -> Self {
        self.user_template = template.into();
        self
    }

    /// Set format
    pub fn format(mut self, fmt: PromptFormat) -> Self {
        self.format = fmt;
        self
    }

    /// Set few-shot config
    pub fn few_shot(mut self, config: FewShotConfig) -> Self {
        self.few_shot = Some(config);
        self
    }

    /// Set reasoning type
    pub fn reasoning(mut self, reasoning: ReasoningType) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Build the prompt with given variables
    pub fn build(&self, variables: &std::collections::HashMap<String, String>) -> String {
        let mut prompt = String::new();

        // System prompt
        prompt.push_str(self.format.system_prefix());
        prompt.push_str(&self.system);
        prompt.push_str(self.format.system_suffix());

        // Few-shot examples
        if let Some(ref fs) = self.few_shot {
            prompt.push_str(&fs.format_examples());
            prompt.push_str(&fs.separator);
        }

        // User prompt with variable substitution
        let mut user = self.user_template.clone();
        for (key, value) in variables {
            user = user.replace(&format!("{{{}}}", key), value);
        }
        prompt.push_str(&user);

        // Reasoning suffix
        prompt.push_str(self.reasoning.prompt_suffix());

        prompt
    }

    /// Estimate token count (rough approximation)
    pub fn estimate_tokens(&self) -> usize {
        let base_tokens = self.system.len() / 4 + self.user_template.len() / 4;
        let few_shot_tokens = self
            .few_shot
            .as_ref()
            .map(|fs| fs.format_examples().len() / 4)
            .unwrap_or(0);
        base_tokens + few_shot_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_format_default() {
        assert_eq!(PromptFormat::default(), PromptFormat::Plain);
    }

    #[test]
    fn test_prompt_format_as_str() {
        assert_eq!(PromptFormat::Plain.as_str(), "plain");
        assert_eq!(PromptFormat::ChatML.as_str(), "chatml");
        assert_eq!(PromptFormat::Alpaca.as_str(), "alpaca");
    }

    #[test]
    fn test_prompt_format_from_str() {
        assert_eq!(PromptFormat::parse("chatml"), Some(PromptFormat::ChatML));
        assert_eq!(PromptFormat::parse("llama"), Some(PromptFormat::Alpaca));
        assert_eq!(PromptFormat::parse("unknown"), None);
    }

    #[test]
    fn test_prompt_format_list_all() {
        assert_eq!(PromptFormat::list_all().len(), 5);
    }

    #[test]
    fn test_prompt_format_prefixes() {
        assert!(PromptFormat::ChatML.system_prefix().contains("im_start"));
        assert!(PromptFormat::Alpaca.system_prefix().contains("Instruction"));
    }

    #[test]
    fn test_few_shot_strategy_default() {
        assert_eq!(FewShotStrategy::default(), FewShotStrategy::Random);
    }

    #[test]
    fn test_few_shot_strategy_as_str() {
        assert_eq!(FewShotStrategy::Random.as_str(), "random");
        assert_eq!(FewShotStrategy::Similar.as_str(), "similar");
        assert_eq!(FewShotStrategy::Diverse.as_str(), "diverse");
    }

    #[test]
    fn test_few_shot_strategy_from_str() {
        assert_eq!(FewShotStrategy::parse("similarity"), Some(FewShotStrategy::Similar));
        assert_eq!(FewShotStrategy::parse("manual"), Some(FewShotStrategy::Fixed));
    }

    #[test]
    fn test_few_shot_strategy_list_all() {
        assert_eq!(FewShotStrategy::list_all().len(), 4);
    }

    #[test]
    fn test_reasoning_type_default() {
        assert_eq!(ReasoningType::default(), ReasoningType::None);
    }

    #[test]
    fn test_reasoning_type_as_str() {
        assert_eq!(ReasoningType::StepByStep.as_str(), "step_by_step");
        assert_eq!(ReasoningType::SelfConsistency.as_str(), "self_consistency");
    }

    #[test]
    fn test_reasoning_type_from_str() {
        assert_eq!(ReasoningType::parse("cot"), Some(ReasoningType::StepByStep));
        assert_eq!(ReasoningType::parse("tot"), Some(ReasoningType::TreeOfThoughts));
    }

    #[test]
    fn test_reasoning_type_prompt_suffix() {
        assert!(ReasoningType::LetThink.prompt_suffix().contains("step by step"));
        assert_eq!(ReasoningType::None.prompt_suffix(), "");
    }

    #[test]
    fn test_few_shot_example_new() {
        let ex = FewShotExample::new("What is 2+2?", "4");
        assert_eq!(ex.input, "What is 2+2?");
        assert_eq!(ex.output, "4");
        assert!(ex.reasoning.is_none());
    }

    #[test]
    fn test_few_shot_example_with_reasoning() {
        let ex = FewShotExample::new("2+2", "4")
            .with_reasoning("2+2 equals 4");
        assert_eq!(ex.reasoning, Some("2+2 equals 4".to_string()));
    }

    #[test]
    fn test_few_shot_example_format() {
        let ex = FewShotExample::new("Q", "A");
        let formatted = ex.format("Input: ", "Output: ");
        assert!(formatted.contains("Input: Q"));
        assert!(formatted.contains("Output: A"));
    }

    #[test]
    fn test_few_shot_config_default() {
        let config = FewShotConfig::default();
        assert!(config.examples.is_empty());
        assert_eq!(config.strategy, FewShotStrategy::Random);
        assert_eq!(config.max_examples, 3);
    }

    #[test]
    fn test_few_shot_config_builder() {
        let config = FewShotConfig::new()
            .example(FewShotExample::new("1+1", "2"))
            .example(FewShotExample::new("2+2", "4"))
            .strategy(FewShotStrategy::Fixed)
            .max_examples(5)
            .separator("\n---\n")
            .input_prefix("Q: ")
            .output_prefix("A: ");

        assert_eq!(config.examples.len(), 2);
        assert_eq!(config.strategy, FewShotStrategy::Fixed);
        assert_eq!(config.max_examples, 5);
        assert_eq!(config.separator, "\n---\n");
    }

    #[test]
    fn test_few_shot_config_format_examples() {
        let config = FewShotConfig::new()
            .example(FewShotExample::new("hello", "world"))
            .input_prefix("In: ")
            .output_prefix("Out: ");

        let formatted = config.format_examples();
        assert!(formatted.contains("In: hello"));
        assert!(formatted.contains("Out: world"));
    }

    #[test]
    fn test_few_shot_config_validate() {
        let valid = FewShotConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = FewShotConfig::new().max_examples(0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_prompt_template_default() {
        let template = PromptTemplate::default();
        assert!(template.system.contains("helpful"));
        assert_eq!(template.user_template, "{input}");
    }

    #[test]
    fn test_prompt_template_builder() {
        let template = PromptTemplate::new()
            .system("You are a math tutor.")
            .user_template("Solve: {problem}")
            .format(PromptFormat::ChatML)
            .reasoning(ReasoningType::StepByStep);

        assert_eq!(template.system, "You are a math tutor.");
        assert_eq!(template.user_template, "Solve: {problem}");
        assert_eq!(template.format, PromptFormat::ChatML);
        assert_eq!(template.reasoning, ReasoningType::StepByStep);
    }

    #[test]
    fn test_prompt_template_build() {
        let template = PromptTemplate::new()
            .system("Assistant")
            .user_template("Hello {name}!")
            .format(PromptFormat::Plain);

        let mut vars = std::collections::HashMap::new();
        vars.insert("name".to_string(), "World".to_string());

        let prompt = template.build(&vars);
        assert!(prompt.contains("Assistant"));
        assert!(prompt.contains("Hello World!"));
    }

    #[test]
    fn test_prompt_template_build_with_few_shot() {
        let fs = FewShotConfig::new()
            .example(FewShotExample::new("1", "one"));

        let template = PromptTemplate::new()
            .system("Translate numbers")
            .user_template("{num}")
            .few_shot(fs);

        let mut vars = std::collections::HashMap::new();
        vars.insert("num".to_string(), "2".to_string());

        let prompt = template.build(&vars);
        assert!(prompt.contains("Translate numbers"));
        assert!(prompt.contains("Input: 1"));
        assert!(prompt.contains("Output: one"));
        assert!(prompt.contains("2"));
    }

    #[test]
    fn test_prompt_template_estimate_tokens() {
        let template = PromptTemplate::new()
            .system("Short")
            .user_template("{x}");

        let tokens = template.estimate_tokens();
        assert!(tokens > 0);
    }

    #[test]
    fn test_prompt_template_with_reasoning() {
        let template = PromptTemplate::new()
            .reasoning(ReasoningType::LetThink);

        let prompt = template.build(&std::collections::HashMap::new());
        assert!(prompt.contains("step by step"));
    }
}
