# AI Agents Example

Run with: `cargo run --example agents_demo`

## Overview

Demonstrates AI agent patterns including:
- Tool definitions
- Memory buffers
- Planning strategies
- Task execution

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::agents::{
    AgentConfig, ToolDefinition, ToolParameter, ParamType, ToolType,
    BufferConfig, MemoryType, ConversationMessage,
    PlanConfig, PlanStep, PlanningStrategy, TaskStatus,
};

fn main() {
    // Agent configuration
    let agent = AgentConfig::default();
    println!("max_iterations: {}", agent.max_iterations);
    println!("tool_timeout_ms: {}", agent.tool_timeout_ms);

    // Define a tool
    let search_tool = ToolDefinition::new("web_search", "Search the web")
        .tool_type(ToolType::Retrieval)
        .parameter(
            ToolParameter::new("query", "Search query")
                .param_type(ParamType::String)
                .required(true)
        );

    // Memory buffer
    let buffer = BufferConfig::default();
    println!("max_messages: {}", buffer.max_messages);
    println!("max_tokens: {}", buffer.max_tokens);

    // Planning
    let plan_config = PlanConfig::default();
    let steps = vec![
        PlanStep::new("step1", "Search for information"),
        PlanStep::new("step2", "Analyze results"),
        PlanStep::new("step3", "Write summary"),
    ];
}
```

## Tool Types

- `Function` - Function calls
- `Retrieval` - Search/retrieval
- `Code` - Code execution
- `Api` - API calls
- `FileSystem` - File operations
- `Database` - Database queries

## Planning Strategies

- `Sequential` - Execute steps in order
- `ReAct` - Reasoning + Acting pattern
- `PlanAndExecute` - Plan first, then execute
- `TreeOfThoughts` - Tree-based exploration
- `ChainOfThought` - Step-by-step reasoning

## Memory Types

- `Buffer` - Simple message buffer
- `Window` - Sliding window
- `Summary` - Summarized context
- `Entity` - Entity-based memory
- `VectorStore` - Vector similarity retrieval
