# Generation API

## Module: `batuta_ground_truth_mlops_corpus::generation`

### SamplingConfig

```rust
pub struct SamplingConfig {
    pub strategy: SamplingStrategy,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub max_new_tokens: usize,
    pub min_new_tokens: usize,
    pub seed: Option<u64>,
}

impl SamplingConfig {
    pub fn new() -> Self;
    pub fn strategy(self, s: SamplingStrategy) -> Self;
    pub fn temperature(self, t: f64) -> Self;
    pub fn top_k(self, k: usize) -> Self;
    pub fn top_p(self, p: f64) -> Self;
    pub fn max_new_tokens(self, n: usize) -> Self;
}
```

### SamplingStrategy

```rust
pub enum SamplingStrategy {
    Greedy,
    Temperature,
    TopK,
    TopP,
    TopKTopP,
    BeamSearch,
    Contrastive,
}
```

### ChatMessage

```rust
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self;
    pub fn user(content: &str) -> Self;
    pub fn assistant(content: &str) -> Self;
}
```

### Conversation

```rust
pub struct Conversation {
    pub messages: Vec<ChatMessage>,
}

impl Conversation {
    pub fn new() -> Self;
    pub fn add_system(&mut self, content: &str);
    pub fn add_user(&mut self, content: &str);
    pub fn add_assistant(&mut self, content: &str);
    pub fn len(&self) -> usize;
}
```
