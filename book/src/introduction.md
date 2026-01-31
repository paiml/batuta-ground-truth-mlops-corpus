# Batuta Ground Truth MLOps Corpus

A production-ready Rust crate providing ground truth MLOps patterns for the Sovereign AI Stack.

## Overview

This corpus provides well-tested, production-ready implementations of common MLOps patterns:

- **Preprocessing**: Tokenization, text processing, feature engineering
- **Models**: Random Forest, Gradient Boosting, Transformers configurations
- **Training**: Trainer, callbacks, learning rate schedulers
- **Evaluation**: Classification/regression metrics, cross-validation
- **Generation**: Sampling strategies, chat templates, prompting
- **RAG**: Chunking, retrieval, reranking, fusion methods
- **Agents**: Tools, memory, planning strategies
- **Safety**: Content filtering, guardrails, privacy, watermarking
- **Hub**: Model registry, versioning, dataset management
- **Deployment**: Quantization, export formats

## Design Principles

### Zero External Dependencies

The corpus uses only Sovereign AI Stack dependencies:

- `trueno` - SIMD/GPU compute primitives
- `aprender` - ML algorithms and APR format

### PMAT A+ Compliance

All code maintains:
- 95%+ test coverage
- 80%+ mutation score
- Zero clippy warnings

### Production-Ready Patterns

Each module provides:
- Builder pattern APIs for configuration
- Comprehensive validation
- Clear error messages
- Full test coverage

## Quick Example

```rust
use batuta_ground_truth_mlops_corpus::training::{
    TrainerConfig, Trainer, TrainingMetrics,
};

let config = TrainerConfig::default()
    .epochs(20)
    .batch_size(32)
    .learning_rate(0.001);

let mut trainer = Trainer::new(config);

// Record training progress
let metrics = TrainingMetrics {
    epoch: 1,
    train_loss: 0.5,
    val_loss: Some(0.6),
    ..Default::default()
};
trainer.record_epoch(metrics);
```

## Running Examples

All examples can be run with `cargo run --example`:

```bash
cargo run --example tokenization_demo
cargo run --example training_demo
cargo run --example generation_demo
cargo run --example rag_demo
cargo run --example agents_demo
cargo run --example safety_demo
cargo run --example hub_demo
cargo run --example evaluation_demo
cargo run --example random_forest_demo
```
