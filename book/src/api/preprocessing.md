# Preprocessing API

## Module: `batuta_ground_truth_mlops_corpus::preprocessing`

### Tokenizer

```rust
pub struct Tokenizer { ... }

impl Tokenizer {
    pub fn whitespace() -> Self;
    pub fn wordpiece() -> Self;
    pub fn bpe() -> Self;
    pub fn with_config(config: TokenizerConfig) -> Self;
    pub fn tokenize(&self, text: &str) -> Vec<Token>;
}
```

### TokenizerConfig

```rust
pub struct TokenizerConfig {
    pub lowercase: bool,
    pub max_length: usize,
    pub truncation: bool,
    pub padding: bool,
    pub pad_token: String,
}

impl TokenizerConfig {
    pub fn new() -> Self;
    pub fn lowercase(self, v: bool) -> Self;
    pub fn max_length(self, n: usize) -> Self;
    pub fn truncation(self, v: bool) -> Self;
}
```

### Token

```rust
pub struct Token {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub token_type: TokenType,
}
```

### TokenType

```rust
pub enum TokenType {
    Word,
    Subword,
    Special,
    Padding,
    Unknown,
}
```
