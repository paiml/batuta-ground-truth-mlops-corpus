//! Agent Planning
//!
//! Task planning, execution strategies, and ReAct patterns.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.agents import PlanningStrategy, TaskStatus, create_plan_config
//! config = create_plan_config(strategy=PlanningStrategy.REACT)
//! ```

/// Planning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanningStrategy {
    /// ReAct (Reasoning + Acting) - default
    #[default]
    ReAct,
    /// Plan and Execute
    PlanAndExecute,
    /// Tree of Thoughts
    TreeOfThoughts,
    /// Chain of Thought
    ChainOfThought,
    /// Sequential
    Sequential,
}

impl PlanningStrategy {
    /// Get strategy name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReAct => "react",
            Self::PlanAndExecute => "plan_and_execute",
            Self::TreeOfThoughts => "tree_of_thoughts",
            Self::ChainOfThought => "chain_of_thought",
            Self::Sequential => "sequential",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "react" => Some(Self::ReAct),
            "plan_and_execute" | "plan" => Some(Self::PlanAndExecute),
            "tree_of_thoughts" | "tot" => Some(Self::TreeOfThoughts),
            "chain_of_thought" | "cot" => Some(Self::ChainOfThought),
            "sequential" | "seq" => Some(Self::Sequential),
            _ => None,
        }
    }

    /// List all strategies
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::ReAct,
            Self::PlanAndExecute,
            Self::TreeOfThoughts,
            Self::ChainOfThought,
            Self::Sequential,
        ]
    }
}

/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskStatus {
    /// Pending execution (default)
    #[default]
    Pending,
    /// Currently running
    Running,
    /// Completed successfully
    Completed,
    /// Failed
    Failed,
    /// Skipped
    Skipped,
    /// Blocked by dependencies
    Blocked,
}

impl TaskStatus {
    /// Get status name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Blocked => "blocked",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "running" | "in_progress" => Some(Self::Running),
            "completed" | "done" => Some(Self::Completed),
            "failed" | "error" => Some(Self::Failed),
            "skipped" | "skip" => Some(Self::Skipped),
            "blocked" => Some(Self::Blocked),
            _ => None,
        }
    }

    /// List all statuses
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Pending,
            Self::Running,
            Self::Completed,
            Self::Failed,
            Self::Skipped,
            Self::Blocked,
        ]
    }

    /// Check if terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Skipped)
    }
}

/// Plan configuration
#[derive(Debug, Clone)]
pub struct PlanConfig {
    /// Planning strategy
    pub strategy: PlanningStrategy,
    /// Maximum iterations/steps
    pub max_iterations: usize,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Allow replanning
    pub allow_replan: bool,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            strategy: PlanningStrategy::ReAct,
            max_iterations: 10,
            timeout_secs: 300,
            allow_replan: true,
            verbose: false,
        }
    }
}

impl PlanConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set strategy
    pub fn strategy(mut self, s: PlanningStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Set max iterations
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set timeout
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set allow replan
    pub fn allow_replan(mut self, enabled: bool) -> Self {
        self.allow_replan = enabled;
        self
    }

    /// Set verbose
    pub fn verbose(mut self, enabled: bool) -> Self {
        self.verbose = enabled;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.max_iterations == 0 {
            return Err("Max iterations must be > 0".to_string());
        }
        Ok(())
    }
}

/// A single plan step
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Step ID
    pub id: String,
    /// Step description
    pub description: String,
    /// Tool to use (if any)
    pub tool: Option<String>,
    /// Tool arguments
    pub arguments: std::collections::HashMap<String, String>,
    /// Status
    pub status: TaskStatus,
    /// Dependencies (step IDs)
    pub dependencies: Vec<String>,
    /// Result (if completed)
    pub result: Option<String>,
}

impl PlanStep {
    /// Create new step
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            tool: None,
            arguments: std::collections::HashMap::new(),
            status: TaskStatus::Pending,
            dependencies: Vec::new(),
            result: None,
        }
    }

    /// Set tool
    pub fn tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }

    /// Add argument
    pub fn argument(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.arguments.insert(key.into(), value.into());
        self
    }

    /// Add dependency
    pub fn depends_on(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Set status
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }

    /// Set result
    pub fn with_result(mut self, result: impl Into<String>) -> Self {
        self.result = Some(result.into());
        self
    }

    /// Check if step can run
    pub fn can_run(&self, completed: &[String]) -> bool {
        self.status == TaskStatus::Pending
            && self.dependencies.iter().all(|d| completed.contains(d))
    }
}

/// Task node in execution graph
#[derive(Debug, Clone)]
pub struct TaskNode {
    /// Node ID
    pub id: String,
    /// Task name
    pub name: String,
    /// Status
    pub status: TaskStatus,
    /// Children task IDs
    pub children: Vec<String>,
    /// Estimated complexity (1-10)
    pub complexity: u8,
}

impl TaskNode {
    /// Create new node
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            status: TaskStatus::Pending,
            children: Vec::new(),
            complexity: 5,
        }
    }

    /// Add child
    pub fn child(mut self, child_id: impl Into<String>) -> Self {
        self.children.push(child_id.into());
        self
    }

    /// Set complexity
    pub fn complexity(mut self, c: u8) -> Self {
        self.complexity = c.min(10);
        self
    }

    /// Set status
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }

    /// Validate node
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Task ID cannot be empty".to_string());
        }
        if self.name.is_empty() {
            return Err("Task name cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Execution plan
#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    /// Plan steps
    pub steps: Vec<PlanStep>,
    /// Plan goal
    pub goal: String,
    /// Strategy used
    pub strategy: PlanningStrategy,
}

impl ExecutionPlan {
    /// Create new plan
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            goal: goal.into(),
            strategy: PlanningStrategy::default(),
        }
    }

    /// Set strategy
    pub fn strategy(mut self, s: PlanningStrategy) -> Self {
        self.strategy = s;
        self
    }

    /// Add step
    pub fn step(mut self, step: PlanStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Get completed step IDs
    pub fn completed_steps(&self) -> Vec<String> {
        self.steps
            .iter()
            .filter(|s| s.status == TaskStatus::Completed)
            .map(|s| s.id.clone())
            .collect()
    }

    /// Get next runnable steps
    pub fn next_steps(&self) -> Vec<&PlanStep> {
        let completed = self.completed_steps();
        self.steps.iter().filter(|s| s.can_run(&completed)).collect()
    }

    /// Count steps by status
    pub fn count_by_status(&self, status: TaskStatus) -> usize {
        self.steps.iter().filter(|s| s.status == status).count()
    }

    /// Check if plan is complete
    pub fn is_complete(&self) -> bool {
        self.steps.iter().all(|s| s.status.is_terminal())
    }
}

/// Planning statistics
#[derive(Debug, Clone, Default)]
pub struct PlanningStats {
    /// Total steps
    pub total_steps: usize,
    /// Completed steps
    pub completed_steps: usize,
    /// Failed steps
    pub failed_steps: usize,
    /// Total iterations
    pub iterations: usize,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl PlanningStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate from plan
    pub fn from_plan(plan: &ExecutionPlan) -> Self {
        Self {
            total_steps: plan.steps.len(),
            completed_steps: plan.count_by_status(TaskStatus::Completed),
            failed_steps: plan.count_by_status(TaskStatus::Failed),
            ..Default::default()
        }
    }

    /// Set iterations
    pub fn iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Set elapsed time
    pub fn elapsed_ms(mut self, ms: u64) -> Self {
        self.elapsed_ms = ms;
        self
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Plan: {}/{} steps completed ({} failed) in {}ms",
            self.completed_steps, self.total_steps, self.failed_steps, self.elapsed_ms
        )
    }
}

/// Calculate plan progress as percentage
pub fn calculate_plan_progress(plan: &ExecutionPlan) -> f64 {
    if plan.steps.is_empty() {
        return 100.0;
    }
    let completed = plan.count_by_status(TaskStatus::Completed) as f64;
    (completed / plan.steps.len() as f64) * 100.0
}

/// Estimate plan complexity (1-100)
pub fn estimate_plan_complexity(steps: usize, max_depth: usize, has_cycles: bool) -> u32 {
    let base = (steps * 5).min(50) as u32;
    let depth_factor = (max_depth * 10).min(30) as u32;
    let cycle_factor = if has_cycles { 20 } else { 0 };
    (base + depth_factor + cycle_factor).min(100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planning_strategy_default() {
        assert_eq!(PlanningStrategy::default(), PlanningStrategy::ReAct);
    }

    #[test]
    fn test_planning_strategy_as_str() {
        assert_eq!(PlanningStrategy::ReAct.as_str(), "react");
        assert_eq!(PlanningStrategy::TreeOfThoughts.as_str(), "tree_of_thoughts");
    }

    #[test]
    fn test_planning_strategy_parse() {
        assert_eq!(PlanningStrategy::parse("react"), Some(PlanningStrategy::ReAct));
        assert_eq!(PlanningStrategy::parse("tot"), Some(PlanningStrategy::TreeOfThoughts));
        assert_eq!(PlanningStrategy::parse("cot"), Some(PlanningStrategy::ChainOfThought));
        assert_eq!(PlanningStrategy::parse("unknown"), None);
    }

    #[test]
    fn test_planning_strategy_list_all() {
        assert_eq!(PlanningStrategy::list_all().len(), 5);
    }

    #[test]
    fn test_task_status_default() {
        assert_eq!(TaskStatus::default(), TaskStatus::Pending);
    }

    #[test]
    fn test_task_status_as_str() {
        assert_eq!(TaskStatus::Pending.as_str(), "pending");
        assert_eq!(TaskStatus::Running.as_str(), "running");
        assert_eq!(TaskStatus::Completed.as_str(), "completed");
    }

    #[test]
    fn test_task_status_parse() {
        assert_eq!(TaskStatus::parse("done"), Some(TaskStatus::Completed));
        assert_eq!(TaskStatus::parse("in_progress"), Some(TaskStatus::Running));
        assert_eq!(TaskStatus::parse("error"), Some(TaskStatus::Failed));
    }

    #[test]
    fn test_task_status_is_terminal() {
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(TaskStatus::Skipped.is_terminal());
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::Running.is_terminal());
    }

    #[test]
    fn test_plan_config_default() {
        let config = PlanConfig::default();
        assert_eq!(config.strategy, PlanningStrategy::ReAct);
        assert_eq!(config.max_iterations, 10);
        assert!(config.allow_replan);
    }

    #[test]
    fn test_plan_config_builder() {
        let config = PlanConfig::new()
            .strategy(PlanningStrategy::TreeOfThoughts)
            .max_iterations(20)
            .timeout_secs(600)
            .allow_replan(false)
            .verbose(true);

        assert_eq!(config.strategy, PlanningStrategy::TreeOfThoughts);
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.timeout_secs, 600);
        assert!(!config.allow_replan);
        assert!(config.verbose);
    }

    #[test]
    fn test_plan_config_validate() {
        let valid = PlanConfig::default();
        assert!(valid.validate().is_ok());

        let zero_iter = PlanConfig::new().max_iterations(0);
        assert!(zero_iter.validate().is_err());
    }

    #[test]
    fn test_plan_step_new() {
        let step = PlanStep::new("step1", "Do something");
        assert_eq!(step.id, "step1");
        assert_eq!(step.description, "Do something");
        assert_eq!(step.status, TaskStatus::Pending);
    }

    #[test]
    fn test_plan_step_builder() {
        let step = PlanStep::new("s1", "Search")
            .tool("search_tool")
            .argument("query", "rust")
            .depends_on("s0")
            .with_status(TaskStatus::Completed)
            .with_result("Found 10 results");

        assert_eq!(step.tool, Some("search_tool".to_string()));
        assert_eq!(step.arguments.get("query"), Some(&"rust".to_string()));
        assert_eq!(step.dependencies, vec!["s0"]);
        assert_eq!(step.status, TaskStatus::Completed);
        assert_eq!(step.result, Some("Found 10 results".to_string()));
    }

    #[test]
    fn test_plan_step_can_run() {
        let step = PlanStep::new("s1", "test").depends_on("s0");
        assert!(!step.can_run(&[]));
        assert!(step.can_run(&["s0".to_string()]));

        let completed = PlanStep::new("s2", "done").with_status(TaskStatus::Completed);
        assert!(!completed.can_run(&[]));
    }

    #[test]
    fn test_task_node_new() {
        let node = TaskNode::new("t1", "Task 1");
        assert_eq!(node.id, "t1");
        assert_eq!(node.name, "Task 1");
        assert_eq!(node.complexity, 5);
    }

    #[test]
    fn test_task_node_builder() {
        let node = TaskNode::new("t1", "Root")
            .child("t2")
            .child("t3")
            .complexity(8)
            .with_status(TaskStatus::Running);

        assert_eq!(node.children.len(), 2);
        assert_eq!(node.complexity, 8);
        assert_eq!(node.status, TaskStatus::Running);
    }

    #[test]
    fn test_task_node_validate() {
        let valid = TaskNode::new("t1", "test");
        assert!(valid.validate().is_ok());

        let no_id = TaskNode::new("", "test");
        assert!(no_id.validate().is_err());

        let no_name = TaskNode::new("t1", "");
        assert!(no_name.validate().is_err());
    }

    #[test]
    fn test_execution_plan_new() {
        let plan = ExecutionPlan::new("Complete task");
        assert_eq!(plan.goal, "Complete task");
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn test_execution_plan_builder() {
        let plan = ExecutionPlan::new("Goal")
            .strategy(PlanningStrategy::Sequential)
            .step(PlanStep::new("s1", "Step 1"))
            .step(PlanStep::new("s2", "Step 2").depends_on("s1"));

        assert_eq!(plan.strategy, PlanningStrategy::Sequential);
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn test_execution_plan_completed_steps() {
        let plan = ExecutionPlan::new("Goal")
            .step(PlanStep::new("s1", "1").with_status(TaskStatus::Completed))
            .step(PlanStep::new("s2", "2").with_status(TaskStatus::Pending));

        let completed = plan.completed_steps();
        assert_eq!(completed, vec!["s1"]);
    }

    #[test]
    fn test_execution_plan_next_steps() {
        let plan = ExecutionPlan::new("Goal")
            .step(PlanStep::new("s1", "1"))
            .step(PlanStep::new("s2", "2").depends_on("s1"));

        let next = plan.next_steps();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].id, "s1");
    }

    #[test]
    fn test_execution_plan_is_complete() {
        let incomplete = ExecutionPlan::new("Goal")
            .step(PlanStep::new("s1", "1"));
        assert!(!incomplete.is_complete());

        let complete = ExecutionPlan::new("Goal")
            .step(PlanStep::new("s1", "1").with_status(TaskStatus::Completed));
        assert!(complete.is_complete());
    }

    #[test]
    fn test_planning_stats_default() {
        let stats = PlanningStats::default();
        assert_eq!(stats.total_steps, 0);
    }

    #[test]
    fn test_planning_stats_from_plan() {
        let plan = ExecutionPlan::new("Goal")
            .step(PlanStep::new("s1", "1").with_status(TaskStatus::Completed))
            .step(PlanStep::new("s2", "2").with_status(TaskStatus::Failed))
            .step(PlanStep::new("s3", "3"));

        let stats = PlanningStats::from_plan(&plan);
        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.completed_steps, 1);
        assert_eq!(stats.failed_steps, 1);
    }

    #[test]
    fn test_planning_stats_format() {
        let stats = PlanningStats::new()
            .iterations(5)
            .elapsed_ms(1000);
        let formatted = stats.format();
        assert!(formatted.contains("1000ms"));
    }

    #[test]
    fn test_calculate_plan_progress() {
        let empty = ExecutionPlan::new("Goal");
        assert_eq!(calculate_plan_progress(&empty), 100.0);

        let half = ExecutionPlan::new("Goal")
            .step(PlanStep::new("s1", "1").with_status(TaskStatus::Completed))
            .step(PlanStep::new("s2", "2"));
        assert!((calculate_plan_progress(&half) - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_plan_complexity() {
        let simple = estimate_plan_complexity(2, 1, false);
        let complex = estimate_plan_complexity(10, 5, true);
        assert!(simple < complex);
        assert!(complex <= 100);
    }
}
