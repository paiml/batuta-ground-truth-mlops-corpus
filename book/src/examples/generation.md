# Text Generation Example

Run with: `cargo run --example generation_demo`

## Overview

Demonstrates text generation utilities including:
- Sampling configurations
- Chat message handling
- Template formats
- Prompt templates
- Conversation management

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::generation::{
    SamplingConfig, SamplingStrategy,
    ChatMessage, Conversation, ChatTemplateFormat,
    PromptTemplate, ReasoningType,
};

fn main() {
    // Greedy sampling (deterministic)
    let greedy = SamplingConfig::new()
        .strategy(SamplingStrategy::Greedy)
        .temperature(0.0);

    // Creative sampling
    let creative = SamplingConfig::new()
        .strategy(SamplingStrategy::TopP)
        .temperature(0.9)
        .top_p(0.95)
        .top_k(50)
        .max_new_tokens(512);

    // Chat messages
    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is Rust?"),
        ChatMessage::assistant("Rust is a systems programming language."),
    ];

    // Conversation management
    let mut conv = Conversation::new();
    conv.add_system("You are an AI assistant.");
    conv.add_user("Hello!");
    conv.add_assistant("Hi! How can I help you?");

    // Prompt template with reasoning
    let template = PromptTemplate::new()
        .system("You are a math tutor")
        .reasoning(ReasoningType::StepByStep);
}
```

## Sampling Strategies

- `Greedy` - Always pick highest probability token
- `Temperature` - Temperature-based sampling
- `TopK` - Sample from top K tokens
- `TopP` - Nucleus sampling (top-p)
- `BeamSearch` - Beam search decoding
- `Contrastive` - Contrastive search

## Chat Template Formats

- `Raw` - Plain text
- `ChatML` - OpenAI ChatML format
- `Llama` - Llama/Llama 2 format
- `Mistral` - Mistral format
- `Gemma` - Gemma format
