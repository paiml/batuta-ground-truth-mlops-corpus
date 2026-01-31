# Bootstrap Corpus Specification

**Version:** 0.1.0
**Status:** Active Development
**PMAT Target:** A+ Grade (95%+ coverage, 80%+ mutation score)

---

## Overview

The Batuta Ground Truth MLOps Corpus provides production-ready Rust implementations of common MLOps patterns. This corpus serves as a cross-language reference, mapping patterns from `hf-ground-truth-corpus` (Python/HuggingFace) to pure Rust using the Sovereign AI Stack.

### Design Principles

1. **Zero External Dependencies**: Only Sovereign AI Stack crates (trueno, aprender)
2. **PMAT A+ Compliance**: 95%+ test coverage, 80%+ mutation score
3. **No Stubs or SATD**: All code must be fully implemented
4. **Cross-Language Parity**: Rust patterns equivalent to Python hf_gtc patterns

---

## Module Structure

| Module | Purpose | Python Equivalent |
|--------|---------|-------------------|
| `preprocessing` | Text/data preprocessing | `hf_gtc.preprocessing` |
| `models` | Model configurations | `sklearn`, `hf_gtc.models` |
| `training` | Training utilities | `hf_gtc.training` |
| `inference` | Inference pipelines | `hf_gtc.inference` |
| `evaluation` | Metrics and cross-validation | `hf_gtc.evaluation` |
| `deployment` | Export and quantization | `hf_gtc.deployment` |

---

## Dependencies

### Production Dependencies

```toml
[dependencies]
# Sovereign AI Stack ONLY - zero external dependencies
trueno = "0.14"     # SIMD tensor operations
aprender = "0.25"   # ML algorithms
```

### Development Dependencies

```toml
[dev-dependencies]
jugar-probar = "1.0"  # Property-based testing (stack alternative to proptest)
```

---

## Quality Gates

### Coverage Requirements

| Metric | Target | Enforcement |
|--------|--------|-------------|
| Line Coverage | ≥95% | `make coverage` fails if below |
| File Coverage | ≥95% per file | Manual review |
| Mutation Score | ≥80% | `make mutants` |

### Testing Tiers

1. **Tier 1 (ON-SAVE)**: `cargo check && cargo clippy` (<1s)
2. **Tier 2 (PRE-COMMIT)**: `make tier2` (fmt, lint, test-lib)
3. **Tier 3 (PRE-PUSH)**: `make quality-gates` (full suite + coverage)

---

## Implementation Status

### Phase 1: Core Infrastructure ✅

- [x] Project scaffold (Cargo.toml, Makefile)
- [x] Error types without external dependencies
- [x] Module structure

### Phase 2: Preprocessing ✅

- [x] `tokenization.rs` - Whitespace, WordPiece tokenizers
- [x] `normalization.rs` - Text normalization
- [x] `augmentation.rs` - Data augmentation (placeholder)

### Phase 3: Models ✅

- [x] `tree.rs` - DecisionTree, RandomForest configs
- [x] `boosting.rs` - GradientBoosting config
- [x] `transformer.rs` - BERT-style transformer configs

### Phase 4: Training ✅

- [x] `trainer.rs` - Trainer with early stopping
- [x] `callbacks.rs` - EarlyStopping, ModelCheckpoint
- [x] `schedulers.rs` - Learning rate schedulers

### Phase 5: Inference ✅

- [x] `pipeline.rs` - Inference pipeline
- [x] `batch.rs` - Batch processing with padding

### Phase 6: Evaluation ✅

- [x] `metrics.rs` - Classification/regression metrics
- [x] `cross_validation.rs` - K-fold cross-validation

### Phase 7: Deployment ✅

- [x] `export.rs` - Model export (APR, JSON, Binary)
- [x] `quantization.rs` - Int8, Int4, Float16 quantization

### Phase 8: Integration 🔄

- [ ] Integration tests with trueno tensors
- [ ] Property-based tests with jugar-probar
- [ ] Benchmark comparisons

---

## Cross-Language Pattern Mapping

### Tokenization

**Python (hf_gtc):**
```python
from hf_gtc.preprocessing.tokenization import Tokenizer
tokenizer = Tokenizer(vocab_size=30000)
tokens = tokenizer.encode("Hello, world!")
```

**Rust (this corpus):**
```rust
use batuta_ground_truth_mlops_corpus::preprocessing::{Tokenizer, TokenizerConfig};
let tokenizer = Tokenizer::new(TokenizerConfig::default());
let tokens = tokenizer.tokenize("Hello, world!");
```

### Random Forest Configuration

**Python (sklearn):**
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
```

**Rust (this corpus):**
```rust
use batuta_ground_truth_mlops_corpus::models::RandomForestConfig;
let config = RandomForestConfig::default()
    .n_estimators(100)
    .max_depth(Some(10))
    .n_jobs(-1);
```

### Model Quantization

**Python (PyTorch):**
```python
import torch.quantization
quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

**Rust (this corpus):**
```rust
use batuta_ground_truth_mlops_corpus::deployment::{Quantizer, QuantizationConfig, QuantizationType};
let quantizer = Quantizer::new(
    QuantizationConfig::default().quantization_type(QuantizationType::Int8)
);
let quantized = quantizer.quantize(&weights);
```

---

## Usage with Batuta RAG Oracle

This corpus is indexed by Batuta's RAG Oracle for cross-language pattern discovery:

```bash
# Index the corpus
batuta oracle --rag-index

# Query for patterns
batuta oracle --rag "how to quantize a model in rust"
# Returns: deployment/quantization.rs with Int8/Int4/Float16 examples

batuta oracle --rag "sklearn random forest equivalent"
# Returns: models/tree.rs with RandomForestConfig
```

---

## Development Workflow

### Quick Start

```bash
# Clone and build
git clone https://github.com/paiml/batuta-ground-truth-mlops-corpus
cd batuta-ground-truth-mlops-corpus
cargo build

# Run tests
make test

# Run with coverage
make coverage

# Run examples
make examples
```

### Adding New Patterns

1. Create module in appropriate directory
2. Add comprehensive tests (target: 95%+ coverage)
3. Add Python equivalent in docstrings
4. Run `make quality-gates`
5. Update this specification

---

## File Size Limits

To maintain code quality and avoid SATD:

| Metric | Limit | Rationale |
|--------|-------|-----------|
| Lines per file | <500 | Maintainability |
| Functions per module | <20 | Single responsibility |
| Test count per file | ≥10 | Coverage compliance |

---

## Changelog

### v0.1.0 (2026-01-31)

- Initial scaffold with all 6 modules
- 215 passing tests
- Makefile with realizar-style coverage
- Zero external dependencies (trueno + aprender only)
