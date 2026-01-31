//! Generation Module
//!
//! Text generation utilities including sampling, prompting, and chat.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.generation import (
//!     SamplingConfig, SamplingStrategy,
//!     PromptTemplate, FewShotConfig,
//!     ChatMessage, Conversation
//! )
//! ```
//!
//! # Example
//!
//! ```rust
//! use batuta_ground_truth_mlops_corpus::generation::{
//!     SamplingConfig, SamplingStrategy,
//!     PromptTemplate, ReasoningType,
//!     ChatMessage, Conversation
//! };
//!
//! // Sampling config
//! let sampling = SamplingConfig::new()
//!     .strategy(SamplingStrategy::TopP)
//!     .temperature(0.7)
//!     .top_p(0.9);
//! assert!(sampling.validate().is_ok());
//!
//! // Prompt template with CoT
//! let template = PromptTemplate::new()
//!     .system("You are a math tutor")
//!     .reasoning(ReasoningType::StepByStep);
//!
//! // Chat conversation
//! let mut conv = Conversation::new();
//! conv.add_user("What is 2+2?");
//! conv.add_assistant("4");
//! assert_eq!(conv.len(), 2);
//! ```

pub mod sampling;
pub mod prompting;
pub mod chat;

pub use sampling::{
    BeamSearchConfig,
    ContrastiveConfig,
    SamplingConfig,
    SamplingStrategy,
    StoppingCriteria,
    estimate_generation_memory,
};

pub use prompting::{
    FewShotConfig,
    FewShotExample,
    FewShotStrategy,
    PromptFormat,
    PromptTemplate,
    ReasoningType,
};

pub use chat::{
    ChatMessage,
    ChatStats,
    ChatTemplateFormat,
    Conversation,
    ConversationConfig,
    MessageRole,
    TruncationStrategy,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_module_exports() {
        // Verify all types are accessible
        let _ = SamplingStrategy::default();
        let _ = PromptFormat::default();
        let _ = MessageRole::default();
    }

    #[test]
    fn test_sampling_prompting_integration() {
        // Create a complete generation setup
        let sampling = SamplingConfig::new()
            .strategy(SamplingStrategy::TopP)
            .temperature(0.8)
            .max_new_tokens(100);

        let prompt = PromptTemplate::new()
            .system("Answer questions")
            .user_template("{question}");

        assert!(sampling.validate().is_ok());
        assert!(prompt.estimate_tokens() > 0);
    }

    #[test]
    fn test_chat_integration() {
        let mut conv = Conversation::new();
        conv.add_system("You are a helpful assistant");
        conv.add_user("Hello!");
        conv.add_assistant("Hi there! How can I help?");

        let stats = ChatStats::from_conversation(&conv);
        assert_eq!(stats.total_messages, 3);
        assert_eq!(stats.user_messages, 1);
        assert_eq!(stats.assistant_messages, 1);
    }

    #[test]
    fn test_few_shot_with_chat() {
        let fs = FewShotConfig::new()
            .example(FewShotExample::new("2+2", "4"))
            .example(FewShotExample::new("3+3", "6"));

        let template = PromptTemplate::new()
            .system("You are a calculator")
            .few_shot(fs);

        let vars = std::collections::HashMap::new();
        let prompt = template.build(&vars);
        
        assert!(prompt.contains("Input: 2+2"));
        assert!(prompt.contains("Output: 4"));
    }

    #[test]
    fn test_beam_search_config() {
        let beam = BeamSearchConfig::new(5)
            .early_stopping(true)
            .length_penalty(1.2);
        
        assert!(beam.validate().is_ok());
        assert_eq!(beam.num_beams, 5);
    }
}
