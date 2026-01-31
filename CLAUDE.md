# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**batuta-ground-truth-mlops-corpus** provides production-ready Rust MLOps patterns for the Sovereign AI Stack. It mirrors [hf-ground-truth-corpus](https://github.com/paiml/hugging-face-ground-truth-corpus) but implements patterns in pure Rust using trueno, aprender, entrenar, and realizar.

## Purpose

This corpus serves as:
1. **Ground truth** for RAG Oracle cross-language pattern discovery
2. **Reference implementations** for Sovereign AI Stack usage
3. **Test suite** for validating stack component integration
4. **Migration guide** from Python (sklearn, PyTorch, HuggingFace) to Rust

## Build and Test Commands

```bash
# Build
cargo build                    # Debug build
cargo build --release          # Release build

# Test
cargo test                     # All tests
cargo test --lib               # Unit tests only
cargo test preprocessing       # Module-specific tests

# Run examples
cargo run --example tokenization_demo
cargo run --example random_forest_demo
cargo run --example inference_pipeline_demo

# Benchmarks
cargo bench

# Lint
cargo clippy -- -D warnings
cargo fmt --check

# Coverage
cargo llvm-cov --lib
```

## Module Structure

| Module | Purpose | Python Equivalent | Stack Components |
|--------|---------|-------------------|------------------|
| `preprocessing` | Text/data preprocessing | `hf_gtc.preprocessing` | trueno |
| `training` | Training loops, optimization | `hf_gtc.training` | entrenar, aprender |
| `inference` | Model inference pipelines | `hf_gtc.inference` | realizar |
| `models` | Model architectures | `hf_gtc.models` | aprender |
| `evaluation` | Metrics, calibration | `hf_gtc.evaluation` | aprender |
| `deployment` | Quantization, ONNX | `hf_gtc.deployment` | realizar |

## Cross-Language Pattern Mapping

Each module contains cross-references to Python equivalents:

```rust
/// Tokenize text for transformer models.
///
/// # Python Equivalent (hf_gtc)
/// ```python
/// from hf_gtc.preprocessing.tokenization import preprocess_text
/// tokens = preprocess_text(text)
/// ```
///
/// # Rust (Sovereign AI Stack)
/// ```rust
/// use batuta_ground_truth_mlops_corpus::preprocessing::tokenize;
/// let tokens = tokenize(text, &config)?;
/// ```
pub fn tokenize(text: &str, config: &TokenizerConfig) -> Result<Vec<Token>> {
    // Implementation using trueno for SIMD acceleration
}
```

## Quality Standards

- **95%+ test coverage** with property-based testing (proptest)
- **Zero clippy warnings** with `-D warnings`
- **Comprehensive documentation** with examples
- **Cross-references** to Python equivalents in all public APIs

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| trueno | 0.14 | SIMD tensor operations |
| aprender | 0.25 | ML algorithms (RandomForest, GradientBoosting, etc.) |
| entrenar | 0.5 | Training loops, autograd, LoRA |
| realizar | 0.6 | Inference runtime, GGUF/SafeTensors |

## Stack Documentation Search

Query this corpus and the entire Sovereign AI Stack using batuta's RAG Oracle:

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search for Rust ML patterns
batuta oracle --rag "tokenization rust"
batuta oracle --rag "random forest aprender"
batuta oracle --rag "inference pipeline realizar"

# Check index status
batuta oracle --rag-stats
```

The RAG index includes this corpus alongside Python ground truth, enabling cross-language pattern discovery.
