//! AI Agents demo showing tools, memory, and planning

use batuta_ground_truth_mlops_corpus::agents::{
    AgentConfig, ToolDefinition, ToolParameter, ParamType, ToolType,
    BufferConfig, MemoryType, ConversationMessage,
    PlanStep, PlanConfig, PlanningStrategy, TaskStatus,
};

fn main() {
    println!("=== AI Agents Demo ===\n");

    // Agent configuration
    println!("Agent Configuration:");
    let agent = AgentConfig::default();

    println!("  max_iterations: {}", agent.max_iterations);
    println!("  tool_timeout_ms: {}", agent.tool_timeout_ms);

    // Tool definitions
    println!("\n--- Tool Definitions ---");

    let search_tool = ToolDefinition::new("web_search", "Search the web for information")
        .tool_type(ToolType::Retrieval)
        .parameter(ToolParameter::new("query", "Search query string")
            .param_type(ParamType::String)
            .required(true))
        .parameter(ToolParameter::new("num_results", "Number of results to return")
            .param_type(ParamType::Integer)
            .required(false));

    println!("\nWeb Search Tool:");
    println!("  name: {}", search_tool.name);
    println!("  description: {}", search_tool.description);
    println!("  tool_type: {:?}", search_tool.tool_type);
    println!("  parameters: {}", search_tool.parameters.len());

    let calculator = ToolDefinition::new("calculator", "Perform mathematical calculations")
        .tool_type(ToolType::Code)
        .parameter(ToolParameter::new("expression", "Mathematical expression to evaluate")
            .param_type(ParamType::String)
            .required(true));

    println!("\nCalculator Tool:");
    println!("  name: {}", calculator.name);
    println!("  description: {}", calculator.description);

    // Tool types
    println!("\nTool Types:");
    let types = ToolType::list_all();
    for t in &types {
        println!("  - {:?}", t);
    }

    // Memory buffer configuration
    println!("\n--- Memory Configuration ---");
    let buffer = BufferConfig::default();

    println!("  max_messages: {}", buffer.max_messages);
    println!("  max_tokens: {}", buffer.max_tokens);
    println!("  return_messages: {}", buffer.return_messages);

    // Memory types
    println!("\nMemory Types:");
    let mem_types = [
        MemoryType::Buffer,
        MemoryType::Summary,
        MemoryType::Window,
        MemoryType::Entity,
    ];
    for t in &mem_types {
        println!("  - {:?}", t);
    }

    // Conversation messages
    println!("\n--- Conversation Messages ---");
    let messages = [
        ConversationMessage::new("user", "Hello, can you help me?"),
        ConversationMessage::new("assistant", "Of course! What do you need?"),
        ConversationMessage::new("user", "I need to search for ML papers."),
    ];
    for msg in &messages {
        println!("  [{}] {}", msg.role, msg.content);
    }

    // Planning
    println!("\n--- Planning ---");
    let plan_config = PlanConfig::default();
    println!("Plan Configuration:");
    println!("  strategy: {:?}", plan_config.strategy);
    println!("  max_iterations: {}", plan_config.max_iterations);

    // Planning strategies
    println!("\nPlanning Strategies:");
    let strategies = [
        PlanningStrategy::Sequential,
        PlanningStrategy::TreeOfThoughts,
        PlanningStrategy::ReAct,
        PlanningStrategy::PlanAndExecute,
    ];
    for s in &strategies {
        println!("  - {:?}", s);
    }

    // Plan steps
    let steps = vec![
        PlanStep::new("step1", "Search for recent Rust ML libraries"),
        PlanStep::new("step2", "Compare features of top 3 libraries"),
        PlanStep::new("step3", "Write summary report"),
    ];

    println!("\nPlan Steps:");
    for step in &steps {
        println!("  {}. {} [{:?}]", step.id, step.description, step.status);
    }

    // Update step status
    let mut step1 = steps[0].clone();
    step1.status = TaskStatus::Completed;
    step1.result = Some("Found 5 Rust ML libraries".to_string());
    println!("\nAfter completing step 1:");
    println!("  {}. {} [{:?}]", step1.id, step1.description, step1.status);
    println!("  Result: {:?}", step1.result);
}
