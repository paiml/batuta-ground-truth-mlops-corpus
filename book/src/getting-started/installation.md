# Installation

## Prerequisites

- Rust 1.75 or later
- Cargo package manager

## Adding to Your Project

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
batuta-ground-truth-mlops-corpus = "0.1"
```

## Feature Flags

The crate supports optional feature flags:

```toml
[dependencies]
batuta-ground-truth-mlops-corpus = { version = "0.1", features = ["training"] }
```

Available features:
- `training` - Training utilities
- `inference` - Inference utilities

## Building from Source

```bash
git clone https://github.com/paiml/batuta-ground-truth-mlops-corpus
cd batuta-ground-truth-mlops-corpus
cargo build --release
```

## Running Tests

```bash
cargo test
```

## Running Examples

```bash
# List all examples
cargo run --example

# Run a specific example
cargo run --example tokenization_demo
```
