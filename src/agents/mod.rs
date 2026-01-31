//! Agent Module
//!
//! Agentic workflows and tool-augmented language models.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.agents import (
//!     MemoryType, BufferConfig, ConversationMessage,
//!     PlanningStrategy, PlanConfig, ExecutionPlan,
//!     ToolType, ToolDefinition, AgentConfig,
//! )
//! ```
//!
//! # Submodules
//! - `memory`: Agent memory management (buffer, window, summary)
//! - `planning`: Task planning and execution
//! - `tools`: Tool definitions and agent configuration

pub mod memory;
pub mod planning;
pub mod tools;

// Re-export key types
pub use memory::{
    BufferConfig, ConversationMessage, EntityConfig, MemoryStats, MemoryType, SummaryConfig,
    WindowConfig,
};
pub use planning::{
    ExecutionPlan, PlanConfig, PlanStep, PlanningStats, PlanningStrategy, TaskNode, TaskStatus,
};
pub use tools::{
    AgentConfig, ParamType, ReActConfig, ToolCall, ToolDefinition, ToolParameter, ToolResult,
    ToolStats, ToolType,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_re_exports() {
        let _ = MemoryType::default();
        let _ = BufferConfig::default();
        let _ = ConversationMessage::new("user", "Hello");
    }

    #[test]
    fn test_planning_re_exports() {
        let _ = PlanningStrategy::default();
        let _ = PlanConfig::default();
        let _ = TaskStatus::default();
    }

    #[test]
    fn test_tools_re_exports() {
        let _ = ToolType::default();
        let _ = AgentConfig::default();
        let _ = ReActConfig::default();
    }

    #[test]
    fn test_integration_agent_with_memory() {
        let config = AgentConfig::new("test_agent")
            .tool(ToolDefinition::new("search", "Search the web"))
            .max_iterations(5);

        let memory_config = BufferConfig::new().max_messages(50);

        assert_eq!(config.name, "test_agent");
        assert_eq!(config.tools.len(), 1);
        assert_eq!(memory_config.max_messages, 50);
    }

    #[test]
    fn test_integration_planning_with_tools() {
        let tool = ToolDefinition::new("execute", "Execute code")
            .tool_type(ToolType::Code);

        let plan_config = PlanConfig::new()
            .strategy(PlanningStrategy::PlanAndExecute)
            .max_iterations(20);

        assert_eq!(tool.tool_type, ToolType::Code);
        assert_eq!(plan_config.strategy, PlanningStrategy::PlanAndExecute);
    }
}
