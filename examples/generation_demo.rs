//! Text generation demo showing sampling, chat, and prompting

use batuta_ground_truth_mlops_corpus::generation::{
    SamplingConfig, SamplingStrategy,
    ChatMessage, Conversation, MessageRole, ChatTemplateFormat,
    PromptTemplate, ReasoningType,
};

fn main() {
    println!("=== Text Generation Demo ===\n");

    // Sampling configuration
    println!("Sampling Configurations:");

    let greedy = SamplingConfig::new()
        .strategy(SamplingStrategy::Greedy)
        .temperature(0.0);
    println!("\nGreedy Sampling:");
    println!("  strategy: {:?}", greedy.strategy);
    println!("  temperature: {}", greedy.temperature);

    let creative = SamplingConfig::new()
        .strategy(SamplingStrategy::TopP)
        .temperature(0.9)
        .top_p(0.95)
        .top_k(50)
        .max_new_tokens(512);
    println!("\nCreative Sampling:");
    println!("  strategy: {:?}", creative.strategy);
    println!("  temperature: {}", creative.temperature);
    println!("  top_p: {:?}", creative.top_p);
    println!("  top_k: {:?}", creative.top_k);
    println!("  max_new_tokens: {}", creative.max_new_tokens);

    // Chat messages
    println!("\n--- Chat Messages ---");

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is Rust?"),
        ChatMessage::assistant("Rust is a systems programming language."),
        ChatMessage::user("Why use it for ML?"),
    ];

    println!("Messages:");
    for msg in &messages {
        println!("  [{:?}] {}", msg.role, msg.content);
    }

    // Chat template formats
    println!("\n--- Chat Template Formats ---");
    let formats = [
        ChatTemplateFormat::Raw,
        ChatTemplateFormat::ChatML,
        ChatTemplateFormat::Llama,
        ChatTemplateFormat::Mistral,
        ChatTemplateFormat::Gemma,
    ];
    for fmt in &formats {
        println!("  - {:?}", fmt);
    }

    // Prompt templates
    println!("\n--- Prompt Templates ---");

    let cot_template = PromptTemplate::new()
        .system("You are a math tutor")
        .reasoning(ReasoningType::StepByStep);
    println!("Chain-of-Thought Template:");
    println!("  system: {:?}", cot_template.system);
    println!("  reasoning: {:?}", cot_template.reasoning);

    // Conversation management
    println!("\n--- Conversation ---");
    let mut conv = Conversation::new();
    conv.add_system("You are an AI assistant.");
    conv.add_user("Hello!");
    conv.add_assistant("Hi! How can I help you?");
    conv.add_user("What's the weather?");

    println!("Messages in conversation: {}", conv.len());
    for msg in &conv.messages {
        let preview: String = msg.content.chars().take(50).collect();
        println!("  [{:?}] {}", msg.role, preview);
    }

    // Sampling strategies
    println!("\n--- Sampling Strategies ---");
    let strategies = [
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK,
        SamplingStrategy::TopP,
        SamplingStrategy::Temperature,
    ];
    for s in &strategies {
        println!("  - {:?}", s);
    }

    // Message roles
    println!("\n--- Message Roles ---");
    let roles = MessageRole::list_all();
    for role in &roles {
        println!("  - {:?}", role);
    }
}
