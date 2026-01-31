//! Agent Tool Management
//!
//! Tool definitions and execution for agentic workflows.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.agents import ToolType, ToolDefinition, create_tool
//! tool = create_tool(name="search", description="Search the web")
//! ```

use std::collections::HashMap;

/// Tool type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolType {
    /// Function calling tool (default)
    #[default]
    Function,
    /// Information retrieval tool
    Retrieval,
    /// Code execution tool
    Code,
    /// External API tool
    Api,
    /// File system tool
    FileSystem,
    /// Database tool
    Database,
}

impl ToolType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Retrieval => "retrieval",
            Self::Code => "code",
            Self::Api => "api",
            Self::FileSystem => "filesystem",
            Self::Database => "database",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "function" | "func" => Some(Self::Function),
            "retrieval" | "search" => Some(Self::Retrieval),
            "code" | "execute" => Some(Self::Code),
            "api" | "http" => Some(Self::Api),
            "filesystem" | "file" | "fs" => Some(Self::FileSystem),
            "database" | "db" | "sql" => Some(Self::Database),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Function,
            Self::Retrieval,
            Self::Code,
            Self::Api,
            Self::FileSystem,
            Self::Database,
        ]
    }

    /// Check if requires external access
    pub fn requires_network(&self) -> bool {
        matches!(self, Self::Api | Self::Retrieval)
    }
}

/// Parameter type for tool definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParamType {
    /// String parameter (default)
    #[default]
    String,
    /// Integer parameter
    Integer,
    /// Float parameter
    Float,
    /// Boolean parameter
    Boolean,
    /// Array parameter
    Array,
    /// Object parameter
    Object,
}

impl ParamType {
    /// Get JSON Schema type name
    pub fn json_schema_type(&self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Float => "number",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "string" | "str" => Some(Self::String),
            "integer" | "int" => Some(Self::Integer),
            "float" | "number" => Some(Self::Float),
            "boolean" | "bool" => Some(Self::Boolean),
            "array" | "list" => Some(Self::Array),
            "object" | "dict" => Some(Self::Object),
            _ => None,
        }
    }
}

/// Tool parameter definition
#[derive(Debug, Clone)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    pub param_type: ParamType,
    /// Is required
    pub required: bool,
    /// Default value
    pub default: Option<String>,
}

impl ToolParameter {
    /// Create new parameter
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ParamType::default(),
            required: true,
            default: None,
        }
    }

    /// Set parameter type
    pub fn param_type(mut self, t: ParamType) -> Self {
        self.param_type = t;
        self
    }

    /// Set required
    pub fn required(mut self, r: bool) -> Self {
        self.required = r;
        self
    }

    /// Set default value
    pub fn default_value(mut self, v: impl Into<String>) -> Self {
        self.default = Some(v.into());
        self.required = false;
        self
    }
}

/// Tool definition
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool type
    pub tool_type: ToolType,
    /// Parameters
    pub parameters: Vec<ToolParameter>,
    /// Return type description
    pub return_type: String,
    /// Whether tool is enabled
    pub enabled: bool,
}

impl ToolDefinition {
    /// Create new tool definition
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            tool_type: ToolType::default(),
            parameters: Vec::new(),
            return_type: "string".to_string(),
            enabled: true,
        }
    }

    /// Set tool type
    pub fn tool_type(mut self, t: ToolType) -> Self {
        self.tool_type = t;
        self
    }

    /// Add parameter
    pub fn parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set return type
    pub fn return_type(mut self, t: impl Into<String>) -> Self {
        self.return_type = t.into();
        self
    }

    /// Set enabled
    pub fn enabled(mut self, e: bool) -> Self {
        self.enabled = e;
        self
    }

    /// Get required parameters
    pub fn required_params(&self) -> Vec<&ToolParameter> {
        self.parameters.iter().filter(|p| p.required).collect()
    }

    /// Get optional parameters
    pub fn optional_params(&self) -> Vec<&ToolParameter> {
        self.parameters.iter().filter(|p| !p.required).collect()
    }

    /// Validate arguments against parameters
    pub fn validate_args(&self, args: &HashMap<String, String>) -> Result<(), String> {
        for param in self.required_params() {
            if !args.contains_key(&param.name) {
                return Err(format!("Missing required parameter: {}", param.name));
            }
        }
        Ok(())
    }
}

/// Tool call request
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Call ID
    pub id: String,
    /// Tool name
    pub tool_name: String,
    /// Arguments
    pub arguments: HashMap<String, String>,
    /// Timestamp
    pub timestamp: u64,
}

impl ToolCall {
    /// Create new tool call
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            id: generate_call_id(),
            tool_name: tool_name.into(),
            arguments: HashMap::new(),
            timestamp: 0,
        }
    }

    /// Set call ID
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Add argument
    pub fn argument(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.arguments.insert(key.into(), value.into());
        self
    }

    /// Set timestamp
    pub fn timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }
}

/// Tool execution result
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Call ID
    pub call_id: String,
    /// Success flag
    pub success: bool,
    /// Result content
    pub content: String,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time (ms)
    pub execution_time_ms: u64,
}

impl ToolResult {
    /// Create success result
    pub fn success(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            success: true,
            content: content.into(),
            error: None,
            execution_time_ms: 0,
        }
    }

    /// Create error result
    pub fn error(call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            success: false,
            content: String::new(),
            error: Some(error.into()),
            execution_time_ms: 0,
        }
    }

    /// Set execution time
    pub fn execution_time(mut self, ms: u64) -> Self {
        self.execution_time_ms = ms;
        self
    }
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Agent name
    pub name: String,
    /// Available tools
    pub tools: Vec<ToolDefinition>,
    /// Max iterations
    pub max_iterations: usize,
    /// Timeout per tool (ms)
    pub tool_timeout_ms: u64,
    /// Enable parallel tool calls
    pub parallel_tools: bool,
    /// Max parallel calls
    pub max_parallel: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "agent".to_string(),
            tools: Vec::new(),
            max_iterations: 10,
            tool_timeout_ms: 30000,
            parallel_tools: false,
            max_parallel: 3,
        }
    }
}

impl AgentConfig {
    /// Create new agent config
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Add tool
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set max iterations
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set tool timeout
    pub fn tool_timeout_ms(mut self, ms: u64) -> Self {
        self.tool_timeout_ms = ms;
        self
    }

    /// Enable parallel tools
    pub fn parallel_tools(mut self, enabled: bool) -> Self {
        self.parallel_tools = enabled;
        self
    }

    /// Set max parallel
    pub fn max_parallel(mut self, n: usize) -> Self {
        self.max_parallel = n;
        self
    }

    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Get enabled tools
    pub fn enabled_tools(&self) -> Vec<&ToolDefinition> {
        self.tools.iter().filter(|t| t.enabled).collect()
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.max_iterations == 0 {
            return Err("Max iterations must be > 0".to_string());
        }
        if self.parallel_tools && self.max_parallel == 0 {
            return Err("Max parallel must be > 0 when parallel is enabled".to_string());
        }
        Ok(())
    }
}

/// ReAct agent configuration
#[derive(Debug, Clone)]
pub struct ReActConfig {
    /// Base agent config
    pub agent: AgentConfig,
    /// Thought prefix
    pub thought_prefix: String,
    /// Action prefix
    pub action_prefix: String,
    /// Observation prefix
    pub observation_prefix: String,
    /// Final answer prefix
    pub final_prefix: String,
    /// Max retries on error
    pub max_retries: usize,
}

impl Default for ReActConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            thought_prefix: "Thought:".to_string(),
            action_prefix: "Action:".to_string(),
            observation_prefix: "Observation:".to_string(),
            final_prefix: "Final Answer:".to_string(),
            max_retries: 2,
        }
    }
}

impl ReActConfig {
    /// Create new ReAct config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set agent config
    pub fn agent(mut self, agent: AgentConfig) -> Self {
        self.agent = agent;
        self
    }

    /// Set thought prefix
    pub fn thought_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.thought_prefix = prefix.into();
        self
    }

    /// Set action prefix
    pub fn action_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.action_prefix = prefix.into();
        self
    }

    /// Set observation prefix
    pub fn observation_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.observation_prefix = prefix.into();
        self
    }

    /// Set final prefix
    pub fn final_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.final_prefix = prefix.into();
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }

    /// Format thought
    pub fn format_thought(&self, thought: &str) -> String {
        format!("{} {}", self.thought_prefix, thought)
    }

    /// Format action
    pub fn format_action(&self, action: &str) -> String {
        format!("{} {}", self.action_prefix, action)
    }

    /// Format observation
    pub fn format_observation(&self, obs: &str) -> String {
        format!("{} {}", self.observation_prefix, obs)
    }

    /// Format final answer
    pub fn format_final(&self, answer: &str) -> String {
        format!("{} {}", self.final_prefix, answer)
    }
}

/// Tool execution statistics
#[derive(Debug, Clone, Default)]
pub struct ToolStats {
    /// Total calls
    pub total_calls: usize,
    /// Successful calls
    pub successful_calls: usize,
    /// Failed calls
    pub failed_calls: usize,
    /// Total execution time (ms)
    pub total_time_ms: u64,
    /// Calls per tool
    pub calls_by_tool: HashMap<String, usize>,
}

impl ToolStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record call
    pub fn record_call(&mut self, tool_name: &str, success: bool, time_ms: u64) {
        self.total_calls += 1;
        if success {
            self.successful_calls += 1;
        } else {
            self.failed_calls += 1;
        }
        self.total_time_ms += time_ms;
        *self.calls_by_tool.entry(tool_name.to_string()).or_insert(0) += 1;
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.successful_calls as f64 / self.total_calls as f64
    }

    /// Get average execution time
    pub fn avg_time_ms(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.total_time_ms as f64 / self.total_calls as f64
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Tools: {} calls ({} success, {} failed), {:.1}ms avg",
            self.total_calls,
            self.successful_calls,
            self.failed_calls,
            self.avg_time_ms()
        )
    }
}

/// Generate a unique call ID
fn generate_call_id() -> String {
    format!("call_{:016x}", rand_u64())
}

/// Simple random u64 (not cryptographic)
fn rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0) as u64;
    seed.wrapping_mul(6364136223846793005).wrapping_add(1)
}

/// Estimate tool call latency in ms
pub fn estimate_tool_latency(tool_type: ToolType) -> u64 {
    match tool_type {
        ToolType::Function => 10,
        ToolType::Retrieval => 200,
        ToolType::Code => 500,
        ToolType::Api => 1000,
        ToolType::FileSystem => 50,
        ToolType::Database => 100,
    }
}

/// Calculate total estimated time for tool calls
pub fn estimate_total_time(tools: &[ToolType], parallel: bool) -> u64 {
    if tools.is_empty() {
        return 0;
    }
    let times: Vec<u64> = tools.iter().map(|t| estimate_tool_latency(*t)).collect();
    if parallel {
        *times.iter().max().unwrap_or(&0)
    } else {
        times.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_type_default() {
        assert_eq!(ToolType::default(), ToolType::Function);
    }

    #[test]
    fn test_tool_type_as_str() {
        assert_eq!(ToolType::Function.as_str(), "function");
        assert_eq!(ToolType::Retrieval.as_str(), "retrieval");
        assert_eq!(ToolType::Code.as_str(), "code");
        assert_eq!(ToolType::Api.as_str(), "api");
        assert_eq!(ToolType::FileSystem.as_str(), "filesystem");
        assert_eq!(ToolType::Database.as_str(), "database");
    }

    #[test]
    fn test_tool_type_parse() {
        assert_eq!(ToolType::parse("function"), Some(ToolType::Function));
        assert_eq!(ToolType::parse("func"), Some(ToolType::Function));
        assert_eq!(ToolType::parse("search"), Some(ToolType::Retrieval));
        assert_eq!(ToolType::parse("execute"), Some(ToolType::Code));
        assert_eq!(ToolType::parse("http"), Some(ToolType::Api));
        assert_eq!(ToolType::parse("fs"), Some(ToolType::FileSystem));
        assert_eq!(ToolType::parse("sql"), Some(ToolType::Database));
        assert_eq!(ToolType::parse("unknown"), None);
    }

    #[test]
    fn test_tool_type_list_all() {
        assert_eq!(ToolType::list_all().len(), 6);
    }

    #[test]
    fn test_tool_type_requires_network() {
        assert!(ToolType::Api.requires_network());
        assert!(ToolType::Retrieval.requires_network());
        assert!(!ToolType::Function.requires_network());
        assert!(!ToolType::Code.requires_network());
    }

    #[test]
    fn test_param_type_json_schema() {
        assert_eq!(ParamType::String.json_schema_type(), "string");
        assert_eq!(ParamType::Integer.json_schema_type(), "integer");
        assert_eq!(ParamType::Float.json_schema_type(), "number");
        assert_eq!(ParamType::Boolean.json_schema_type(), "boolean");
        assert_eq!(ParamType::Array.json_schema_type(), "array");
        assert_eq!(ParamType::Object.json_schema_type(), "object");
    }

    #[test]
    fn test_param_type_parse() {
        assert_eq!(ParamType::parse("str"), Some(ParamType::String));
        assert_eq!(ParamType::parse("int"), Some(ParamType::Integer));
        assert_eq!(ParamType::parse("number"), Some(ParamType::Float));
        assert_eq!(ParamType::parse("bool"), Some(ParamType::Boolean));
        assert_eq!(ParamType::parse("list"), Some(ParamType::Array));
        assert_eq!(ParamType::parse("dict"), Some(ParamType::Object));
        assert_eq!(ParamType::parse("unknown"), None);
    }

    #[test]
    fn test_tool_parameter_new() {
        let param = ToolParameter::new("query", "Search query");
        assert_eq!(param.name, "query");
        assert_eq!(param.description, "Search query");
        assert!(param.required);
    }

    #[test]
    fn test_tool_parameter_builder() {
        let param = ToolParameter::new("count", "Result count")
            .param_type(ParamType::Integer)
            .required(false)
            .default_value("10");

        assert_eq!(param.param_type, ParamType::Integer);
        assert!(!param.required);
        assert_eq!(param.default, Some("10".to_string()));
    }

    #[test]
    fn test_tool_definition_new() {
        let tool = ToolDefinition::new("search", "Search the web");
        assert_eq!(tool.name, "search");
        assert_eq!(tool.description, "Search the web");
        assert!(tool.enabled);
    }

    #[test]
    fn test_tool_definition_builder() {
        let tool = ToolDefinition::new("search", "Search the web")
            .tool_type(ToolType::Retrieval)
            .parameter(ToolParameter::new("query", "Search query"))
            .parameter(ToolParameter::new("limit", "Max results").required(false))
            .return_type("array")
            .enabled(true);

        assert_eq!(tool.tool_type, ToolType::Retrieval);
        assert_eq!(tool.parameters.len(), 2);
        assert_eq!(tool.return_type, "array");
    }

    #[test]
    fn test_tool_definition_required_params() {
        let tool = ToolDefinition::new("search", "Search")
            .parameter(ToolParameter::new("query", "Query"))
            .parameter(ToolParameter::new("limit", "Limit").required(false));

        assert_eq!(tool.required_params().len(), 1);
        assert_eq!(tool.optional_params().len(), 1);
    }

    #[test]
    fn test_tool_definition_validate_args() {
        let tool = ToolDefinition::new("search", "Search")
            .parameter(ToolParameter::new("query", "Query"));

        let mut args = HashMap::new();
        assert!(tool.validate_args(&args).is_err());

        args.insert("query".to_string(), "test".to_string());
        assert!(tool.validate_args(&args).is_ok());
    }

    #[test]
    fn test_tool_call_new() {
        let call = ToolCall::new("search");
        assert_eq!(call.tool_name, "search");
        assert!(call.id.starts_with("call_"));
    }

    #[test]
    fn test_tool_call_builder() {
        let call = ToolCall::new("search")
            .id("custom_id")
            .argument("query", "test")
            .timestamp(12345);

        assert_eq!(call.id, "custom_id");
        assert_eq!(call.arguments.get("query"), Some(&"test".to_string()));
        assert_eq!(call.timestamp, 12345);
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("call_1", "Found 10 results")
            .execution_time(100);

        assert!(result.success);
        assert_eq!(result.content, "Found 10 results");
        assert!(result.error.is_none());
        assert_eq!(result.execution_time_ms, 100);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("call_1", "Connection failed")
            .execution_time(50);

        assert!(!result.success);
        assert!(result.content.is_empty());
        assert_eq!(result.error, Some("Connection failed".to_string()));
    }

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.name, "agent");
        assert_eq!(config.max_iterations, 10);
        assert!(!config.parallel_tools);
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentConfig::new("my_agent")
            .tool(ToolDefinition::new("search", "Search"))
            .max_iterations(20)
            .tool_timeout_ms(5000)
            .parallel_tools(true)
            .max_parallel(5);

        assert_eq!(config.name, "my_agent");
        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.tool_timeout_ms, 5000);
        assert!(config.parallel_tools);
        assert_eq!(config.max_parallel, 5);
    }

    #[test]
    fn test_agent_config_get_tool() {
        let config = AgentConfig::new("agent")
            .tool(ToolDefinition::new("search", "Search"))
            .tool(ToolDefinition::new("calc", "Calculate"));

        assert!(config.get_tool("search").is_some());
        assert!(config.get_tool("calc").is_some());
        assert!(config.get_tool("unknown").is_none());
    }

    #[test]
    fn test_agent_config_enabled_tools() {
        let config = AgentConfig::new("agent")
            .tool(ToolDefinition::new("search", "Search"))
            .tool(ToolDefinition::new("disabled", "Disabled").enabled(false));

        assert_eq!(config.enabled_tools().len(), 1);
    }

    #[test]
    fn test_agent_config_validate() {
        let valid = AgentConfig::default();
        assert!(valid.validate().is_ok());

        let zero_iter = AgentConfig::new("agent").max_iterations(0);
        assert!(zero_iter.validate().is_err());

        let zero_parallel = AgentConfig::new("agent")
            .parallel_tools(true)
            .max_parallel(0);
        assert!(zero_parallel.validate().is_err());
    }

    #[test]
    fn test_react_config_default() {
        let config = ReActConfig::default();
        assert_eq!(config.thought_prefix, "Thought:");
        assert_eq!(config.action_prefix, "Action:");
        assert_eq!(config.max_retries, 2);
    }

    #[test]
    fn test_react_config_builder() {
        let config = ReActConfig::new()
            .agent(AgentConfig::new("react_agent"))
            .thought_prefix("Think:")
            .action_prefix("Do:")
            .observation_prefix("See:")
            .final_prefix("Answer:")
            .max_retries(3);

        assert_eq!(config.agent.name, "react_agent");
        assert_eq!(config.thought_prefix, "Think:");
        assert_eq!(config.action_prefix, "Do:");
        assert_eq!(config.observation_prefix, "See:");
        assert_eq!(config.final_prefix, "Answer:");
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_react_config_format_methods() {
        let config = ReActConfig::default();

        assert_eq!(config.format_thought("I need to search"), "Thought: I need to search");
        assert_eq!(config.format_action("search[query]"), "Action: search[query]");
        assert_eq!(config.format_observation("Found 5 results"), "Observation: Found 5 results");
        assert_eq!(config.format_final("The answer is 42"), "Final Answer: The answer is 42");
    }

    #[test]
    fn test_tool_stats_new() {
        let stats = ToolStats::new();
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_tool_stats_record_call() {
        let mut stats = ToolStats::new();
        stats.record_call("search", true, 100);
        stats.record_call("search", true, 150);
        stats.record_call("calc", false, 50);

        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.successful_calls, 2);
        assert_eq!(stats.failed_calls, 1);
        assert_eq!(stats.total_time_ms, 300);
        assert_eq!(stats.calls_by_tool.get("search"), Some(&2));
        assert_eq!(stats.calls_by_tool.get("calc"), Some(&1));
    }

    #[test]
    fn test_tool_stats_success_rate() {
        let mut stats = ToolStats::new();
        stats.record_call("a", true, 10);
        stats.record_call("b", true, 10);
        stats.record_call("c", false, 10);
        stats.record_call("d", false, 10);

        assert!((stats.success_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_tool_stats_avg_time() {
        let mut stats = ToolStats::new();
        stats.record_call("a", true, 100);
        stats.record_call("b", true, 200);
        stats.record_call("c", true, 300);

        assert!((stats.avg_time_ms() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_tool_stats_format() {
        let mut stats = ToolStats::new();
        stats.record_call("search", true, 100);
        let formatted = stats.format();

        assert!(formatted.contains("1 calls"));
        assert!(formatted.contains("1 success"));
        assert!(formatted.contains("0 failed"));
    }

    #[test]
    fn test_estimate_tool_latency() {
        assert_eq!(estimate_tool_latency(ToolType::Function), 10);
        assert_eq!(estimate_tool_latency(ToolType::Retrieval), 200);
        assert_eq!(estimate_tool_latency(ToolType::Code), 500);
        assert_eq!(estimate_tool_latency(ToolType::Api), 1000);
        assert_eq!(estimate_tool_latency(ToolType::FileSystem), 50);
        assert_eq!(estimate_tool_latency(ToolType::Database), 100);
    }

    #[test]
    fn test_estimate_total_time_empty() {
        assert_eq!(estimate_total_time(&[], false), 0);
        assert_eq!(estimate_total_time(&[], true), 0);
    }

    #[test]
    fn test_estimate_total_time_sequential() {
        let tools = vec![ToolType::Function, ToolType::Retrieval, ToolType::Api];
        let total = estimate_total_time(&tools, false);
        assert_eq!(total, 10 + 200 + 1000); // Sum
    }

    #[test]
    fn test_estimate_total_time_parallel() {
        let tools = vec![ToolType::Function, ToolType::Retrieval, ToolType::Api];
        let total = estimate_total_time(&tools, true);
        assert_eq!(total, 1000); // Max
    }

    #[test]
    fn test_generate_call_id() {
        let id1 = generate_call_id();
        let id2 = generate_call_id();

        assert!(id1.starts_with("call_"));
        assert!(id2.starts_with("call_"));
    }
}
