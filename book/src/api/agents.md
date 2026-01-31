# Agents API

## Module: `batuta_ground_truth_mlops_corpus::agents`

### AgentConfig

```rust
pub struct AgentConfig {
    pub max_iterations: usize,
    pub tool_timeout_ms: u64,
}

impl AgentConfig {
    pub fn new() -> Self;
    pub fn max_iterations(self, n: usize) -> Self;
    pub fn tool_timeout_ms(self, ms: u64) -> Self;
}
```

### ToolDefinition

```rust
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub tool_type: ToolType,
    pub parameters: Vec<ToolParameter>,
}

impl ToolDefinition {
    pub fn new(name: &str, description: &str) -> Self;
    pub fn tool_type(self, t: ToolType) -> Self;
    pub fn parameter(self, param: ToolParameter) -> Self;
}
```

### ToolParameter

```rust
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub param_type: ParamType,
    pub required: bool,
}

impl ToolParameter {
    pub fn new(name: &str, description: &str) -> Self;
    pub fn param_type(self, t: ParamType) -> Self;
    pub fn required(self, r: bool) -> Self;
}
```

### PlanConfig

```rust
pub struct PlanConfig {
    pub strategy: PlanningStrategy,
    pub max_iterations: usize,
}
```

### PlanStep

```rust
pub struct PlanStep {
    pub id: String,
    pub description: String,
    pub status: TaskStatus,
    pub result: Option<String>,
}

impl PlanStep {
    pub fn new(id: &str, description: &str) -> Self;
}
```
