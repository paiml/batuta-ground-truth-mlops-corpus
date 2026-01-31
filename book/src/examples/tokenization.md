# Tokenization Example

Run with: `cargo run --example tokenization_demo`

## Overview

Demonstrates various tokenization strategies and configurations.

## Code

```rust
use batuta_ground_truth_mlops_corpus::preprocessing::{
    Tokenizer, TokenizerConfig,
};

fn main() {
    // Whitespace tokenization
    let tokenizer = Tokenizer::whitespace();
    let text = "Hello, world! This is a test.";
    let tokens = tokenizer.tokenize(text);

    for token in &tokens {
        println!("  '{}' [{}, {}]", token.text, token.start, token.end);
    }

    // WordPiece tokenization
    let wordpiece = Tokenizer::wordpiece();
    let tokens = wordpiece.tokenize("understanding transformers");

    // Custom configuration
    let config = TokenizerConfig::new()
        .lowercase(true)
        .max_length(5)
        .truncation(true);
    let custom = Tokenizer::with_config(config);
}
```

## Output

```
=== Tokenization Demo ===

Whitespace tokenization of: "Hello, world! This is a test."
  - 'hello,' [0, 6]
  - 'world!' [7, 13]
  - 'this' [14, 18]
  - 'is' [19, 21]
  - 'a' [22, 23]
  - 'test.' [24, 29]

WordPiece tokenization:
  - 'unders' [0, 6]
  - '##tanding' [6, 13]
  - 'transf' [14, 20]
  - '##ormers' [20, 26]
```
