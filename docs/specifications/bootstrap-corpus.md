# Bootstrap Corpus Specification

**Version:** 0.2.0
**Status:** Active Development
**PMAT Target:** A+ Grade (95%+ coverage, 80%+ mutation score)
**Python Reference:** `hf-ground-truth-corpus` (14 modules, 7,800 lines API)

---

## Overview

The Batuta Ground Truth MLOps Corpus provides production-ready Rust implementations of MLOps patterns with full parity to `hf-ground-truth-corpus`. This corpus serves as a cross-language reference, mapping Python/HuggingFace patterns to pure Rust using the Sovereign AI Stack.

### Design Principles

1. **Zero External Dependencies**: Only Sovereign AI Stack crates (trueno, aprender)
2. **PMAT A+ Compliance**: 95%+ test coverage, 80%+ mutation score
3. **No Stubs or SATD**: All code must be fully implemented
4. **Cross-Language Parity**: 1:1 mapping with Python hf_gtc patterns
5. **Depyler Compatible**: Patterns structured for automatic transpilation

---

## Module Parity Matrix

Full parity with `hf-ground-truth-corpus` 14 modules:

| # | Python Module | Rust Module | Status | Files | Priority |
|---|---------------|-------------|--------|-------|----------|
| 1 | `hf_gtc.preprocessing` | `preprocessing` | ✅ Done | 3 | P0 |
| 2 | `hf_gtc.models` | `models` | ✅ Done | 4 | P0 |
| 3 | `hf_gtc.training` | `training` | ✅ Done | 4 | P0 |
| 4 | `hf_gtc.inference` | `inference` | ✅ Done | 3 | P0 |
| 5 | `hf_gtc.evaluation` | `evaluation` | ✅ Done | 3 | P0 |
| 6 | `hf_gtc.deployment` | `deployment` | ✅ Done | 3 | P0 |
| 7 | `hf_gtc.hub` | `hub` | 🔲 Planned | 0 | P1 |
| 8 | `hf_gtc.generation` | `generation` | 🔲 Planned | 0 | P1 |
| 9 | `hf_gtc.rag` | `rag` | 🔲 Planned | 0 | P1 |
| 10 | `hf_gtc.agents` | `agents` | 🔲 Planned | 0 | P2 |
| 11 | `hf_gtc.safety` | `safety` | 🔲 Planned | 0 | P2 |
| 12 | `hf_gtc.audio` | `audio` | 🔲 Planned | 0 | P3 |
| 13 | `hf_gtc.multimodal` | `multimodal` | 🔲 Planned | 0 | P3 |
| 14 | (tests) | `tests` | 🔄 Partial | - | P0 |

---

## Detailed Module Specifications

### 1. Preprocessing (`hf_gtc.preprocessing` → `preprocessing`)

**Python Reference**: 13 files, 747 lines API

| Python File | Rust File | Status | Key Types |
|-------------|-----------|--------|-----------|
| `tokenization.py` | `tokenization.rs` | ✅ | `Token`, `Tokenizer`, `TokenizerConfig`, `TokenizerType` |
| `vocabulary.py` | (in tokenization) | ✅ | Vocabulary analysis embedded |
| `augmentation.py` | `augmentation.rs` | ✅ | `Augmenter`, `AugmentationConfig`, `AugmentationType` |
| `filtering.py` | `normalization.rs` | ✅ | `Normalizer`, `NormalizerConfig` |
| `synthetic.py` | 🔲 Planned | - | Synthetic data generation |
| `sampling.py` | 🔲 Planned | - | Stratified sampling |
| `quality.py` | 🔲 Planned | - | Quality metrics |
| `curation.py` | 🔲 Planned | - | Dataset curation |
| `pipeline.py` | 🔲 Planned | - | Preprocessing pipeline |
| `streaming.py` | 🔲 Planned | - | Streaming data |

**Cross-Language Examples:**

```python
# Python (hf_gtc)
from hf_gtc.preprocessing.tokenization import (
    TokenizerType, VALID_TOKENIZER_TYPES,
    calculate_vocab_fertility, analyze_vocabulary
)
tokenizer_type = TokenizerType.BPE
fertility = calculate_vocab_fertility(tokenizer, corpus)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::preprocessing::{
    TokenizerType, Tokenizer, TokenizerConfig
};
let tokenizer = Tokenizer::new(TokenizerConfig::default()
    .tokenizer_type(TokenizerType::Bpe));
let tokens = tokenizer.tokenize("Hello, world!");
```

---

### 2. Models (`hf_gtc.models` → `models`)

**Python Reference**: 8 files, 593 lines API

| Python File | Rust File | Status | Key Types |
|-------------|-----------|--------|-----------|
| `attention.py` | `transformer.rs` | ✅ | `TransformerConfig` (attention embedded) |
| `positional.py` | (in transformer) | ✅ | Positional encoding types |
| `normalization.py` | (in transformer) | ✅ | Layer normalization |
| `activations.py` | 🔲 Planned | - | GELU, SwiGLU, Mish |
| `layers.py` | 🔲 Planned | - | Transformer layers |
| `architectures.py` | 🔲 Planned | - | Architecture patterns |
| `analysis.py` | 🔲 Planned | - | Model analysis |

**Additional Rust Files (sklearn parity):**

| Rust File | Status | Key Types |
|-----------|--------|-----------|
| `tree.rs` | ✅ | `RandomForestConfig`, `DecisionTreeConfig`, `SplitCriterion`, `MaxFeatures` |
| `boosting.rs` | ✅ | `GradientBoostingConfig` |

**Cross-Language Examples:**

```python
# Python (hf_gtc)
from hf_gtc.models.attention import (
    AttentionType, calculate_attention_complexity
)
complexity = calculate_attention_complexity(seq_len=1024, heads=12, dim=768)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::models::TransformerConfig;
let config = TransformerConfig::bert_base();
assert_eq!(config.num_attention_heads, 12);
assert_eq!(config.hidden_size, 768);
```

---

### 3. Training (`hf_gtc.training` → `training`)

**Python Reference**: 34 files, 1657 lines API (largest module)

| Python File | Rust File | Status | Key Types |
|-------------|-----------|--------|-----------|
| `trainer.py` | `trainer.rs` | ✅ | `Trainer`, `TrainerConfig`, `TrainingMetrics` |
| `callbacks.py` | `callbacks.rs` | ✅ | `EarlyStopping`, `ModelCheckpoint`, `CallbackType` |
| `schedulers.py` | `schedulers.rs` | ✅ | `LearningRateScheduler`, `SchedulerType` |
| `lora.py` | 🔲 Planned | - | LoRA configuration |
| `qlora.py` | 🔲 Planned | - | QLoRA with quantization |
| `fine_tuning.py` | 🔲 Planned | - | Fine-tuning utilities |
| `adapters.py` | 🔲 Planned | - | AdaLoRA, IA3, PrefixTuning |
| `optimizers.py` | 🔲 Planned | - | AdamW, LAMB, SGD configs |
| `gradient.py` | 🔲 Planned | - | Gradient accumulation, clipping |
| `mixed_precision.py` | 🔲 Planned | - | fp16/bf16 training |
| `parallelism.py` | 🔲 Planned | - | Data/model parallelism |
| `active_learning.py` | 🔲 Planned | - | Query strategies |
| `distillation.py` | 🔲 Planned | - | Knowledge distillation |
| `pruning.py` | 🔲 Planned | - | Magnitude pruning |
| `merging.py` | 🔲 Planned | - | TIES, SLERP merging |
| `dpo.py` | 🔲 Planned | - | Direct Preference Optimization |
| `ppo.py` | 🔲 Planned | - | PPO reinforcement learning |
| `checkpointing.py` | 🔲 Planned | - | Checkpoint management |
| `collators.py` | 🔲 Planned | - | Batch collation |
| `losses.py` | 🔲 Planned | - | Custom loss functions |
| `reproducibility.py` | 🔲 Planned | - | Seed management |

**Cross-Language Examples:**

```python
# Python (hf_gtc)
from hf_gtc.training.lora import LoRAConfig, calculate_lora_rank
config = LoRAConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
trainable_params = config.estimate_trainable_params(model_size=7_000_000_000)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::training::{
    Trainer, TrainerConfig, EarlyStopping
};
let trainer = Trainer::new(
    TrainerConfig::default()
        .epochs(20)
        .learning_rate(0.001)
        .early_stopping(5)
);
```

```python
# Python (hf_gtc) - Learning Rate Schedulers
from hf_gtc.training.schedulers import SchedulerType, create_scheduler
scheduler = create_scheduler(SchedulerType.COSINE_ANNEALING, lr=0.001, total_steps=1000)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::training::{LearningRateScheduler, SchedulerType};
let mut scheduler = LearningRateScheduler::cosine_annealing(1.0, 1000, 0.0);
scheduler.step();
```

---

### 4. Inference (`hf_gtc.inference` → `inference`)

**Python Reference**: 17 files, 897 lines API

| Python File | Rust File | Status | Key Types |
|-------------|-----------|--------|-----------|
| `pipelines.py` | `pipeline.rs` | ✅ | `InferencePipeline`, `PipelineConfig`, `PipelineResult` |
| `batch.py` | `batch.rs` | ✅ | `BatchProcessor`, `BatchConfig`, `BatchResult`, `PaddingStrategy` |
| `batching.py` | (in batch) | ✅ | Continuous/dynamic batching |
| `engines.py` | 🔲 Planned | - | ONNX, TFLite, GGUF engines |
| `device.py` | 🔲 Planned | - | Device management |
| `caching.py` | 🔲 Planned | - | Prompt/KV cache |
| `kv_cache.py` | 🔲 Planned | - | KV cache optimization |
| `speculative.py` | 🔲 Planned | - | Speculative decoding |
| `streaming.py` | 🔲 Planned | - | Token streaming |
| `decoding.py` | 🔲 Planned | - | Beam search, nucleus sampling |
| `quantization.py` | 🔲 Planned | - | Runtime quantization |
| `embeddings.py` | 🔲 Planned | - | Embedding pipeline |
| `memory.py` | 🔲 Planned | - | Memory profiling |
| `hardware.py` | 🔲 Planned | - | Hardware detection |

**Cross-Language Examples:**

```python
# Python (hf_gtc)
from hf_gtc.inference.batching import (
    BatchConfig, calculate_optimal_batch_size, PaddingStrategy
)
config = BatchConfig(max_batch_size=32, padding=PaddingStrategy.LONGEST)
optimal = calculate_optimal_batch_size(model_memory=4_000_000_000, gpu_memory=16_000_000_000)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::inference::{
    BatchProcessor, BatchConfig, PaddingStrategy
};
let processor = BatchProcessor::new(
    BatchConfig::default()
        .max_batch_size(32)
        .padding(PaddingStrategy::Longest)
);
```

---

### 5. Evaluation (`hf_gtc.evaluation` → `evaluation`)

**Python Reference**: 13 files, 872 lines API

| Python File | Rust File | Status | Key Types |
|-------------|-----------|--------|-----------|
| `metrics.py` | `metrics.rs` | ✅ | `ClassificationMetrics`, `RegressionMetrics`, `Metrics` |
| `benchmarks.py` | 🔲 Planned | - | GLUE, SuperGLUE, MTEB |
| `harness.py` | 🔲 Planned | - | Evaluation harness |
| `calibration.py` | 🔲 Planned | - | ECE, MCE metrics |
| `comparison.py` | 🔲 Planned | - | Statistical significance |
| `bias.py` | 🔲 Planned | - | Fairness metrics |
| `robustness.py` | 🔲 Planned | - | Adversarial testing |
| `interpretability.py` | 🔲 Planned | - | Feature importance |
| `profiling.py` | 🔲 Planned | - | Latency/throughput |

**Additional Rust Files:**

| Rust File | Status | Key Types |
|-----------|--------|-----------|
| `cross_validation.rs` | ✅ | `CrossValidator`, `CrossValidationConfig`, `FoldResult` |

**Cross-Language Examples:**

```python
# Python (hf_gtc)
from hf_gtc.evaluation.metrics import (
    compute_classification_metrics, compute_regression_metrics,
    calculate_confidence_interval
)
metrics = compute_classification_metrics(y_true, y_pred)
ci = calculate_confidence_interval(metrics.accuracy, n_samples=1000)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::evaluation::{
    ClassificationMetrics, accuracy, f1_score, confusion_matrix
};
let metrics = ClassificationMetrics::compute(&y_true, &y_pred);
let cm = confusion_matrix(&y_true, &y_pred);
```

---

### 6. Deployment (`hf_gtc.deployment` → `deployment`)

**Python Reference**: 13 files, 821 lines API

| Python File | Rust File | Status | Key Types |
|-------------|-----------|--------|-----------|
| `quantization.py` | `quantization.rs` | ✅ | `Quantizer`, `QuantizationConfig`, `QuantizationType`, `QuantizedModel` |
| `safetensors.py` | `export.rs` | ✅ | `Exporter`, `ExportConfig`, `ExportFormat` (APR = SafeTensors equivalent) |
| `gguf.py` | 🔲 Planned | - | GGUF format support |
| `onnx.py` | 🔲 Planned | - | ONNX export |
| `torchscript.py` | 🔲 Planned | - | (N/A for Rust) |
| `tflite.py` | 🔲 Planned | - | TFLite conversion |
| `conversion.py` | 🔲 Planned | - | Format conversion |
| `compression.py` | 🔲 Planned | - | Pruning, distillation |
| `optimization.py` | 🔲 Planned | - | Graph optimization |
| `serving.py` | 🔲 Planned | - | Model serving |
| `merging.py` | 🔲 Planned | - | Model merging |
| `cost.py` | 🔲 Planned | - | Cost estimation |

**Cross-Language Examples:**

```python
# Python (hf_gtc)
from hf_gtc.deployment.quantization import (
    QuantizationType, QuantizationConfig, quantize_model
)
config = QuantizationConfig(dtype=QuantizationType.INT8, per_channel=True)
quantized = quantize_model(model, config)
```

```rust
// Rust (this corpus)
use batuta_ground_truth_mlops_corpus::deployment::{
    Quantizer, QuantizationConfig, QuantizationType
};
let quantizer = Quantizer::new(
    QuantizationConfig::default()
        .quantization_type(QuantizationType::Int8)
        .per_channel(true)
);
let quantized = quantizer.quantize(&weights);
```

---

### 7-14. Planned Modules

#### 7. Hub (`hf_gtc.hub` → `hub`) - P1

**Python Reference**: 11 files, 556 lines API

| Python File | Rust Equivalent | Key Types |
|-------------|-----------------|-----------|
| `search.py` | `search.rs` | `ModelInfo`, `DatasetInfo`, search filters |
| `model_cards.py` | `model_cards.rs` | Model card metadata |
| `datasets.py` | `datasets.rs` | `DatasetConfig`, `DatasetMetadata` |
| `versioning.py` | `versioning.rs` | `ModelVersion`, SHA-256 hashing |
| `registry.py` | `registry.rs` | Registry integration (maps to pacha) |

#### 8. Generation (`hf_gtc.generation` → `generation`) - P1

**Python Reference**: 9 files, 495 lines API

| Python File | Rust Equivalent | Key Types |
|-------------|-----------------|-----------|
| `prompting.py` | `prompting.rs` | Few-shot templates, CoT |
| `chat.py` | `chat.rs` | Chat completion API |
| `sampling.py` | `sampling.rs` | Temperature, top-k, top-p |
| `constraints.py` | `constraints.rs` | Grammar constraints |
| `structured.py` | `structured.rs` | JSON schema output |
| `tools.py` | `tools.rs` | Function calling |

#### 9. RAG (`hf_gtc.rag` → `rag`) - P1

**Python Reference**: 8 files, 467 lines API

| Python File | Rust Equivalent | Key Types |
|-------------|-----------------|-----------|
| `vectorstore.py` | `vectorstore.rs` | Vector DB integration (maps to trueno-db) |
| `chunking.py` | `chunking.rs` | Semantic chunking (maps to trueno-rag) |
| `retrieval.py` | `retrieval.rs` | BM25 + vector search |
| `hybrid_search.py` | `hybrid.rs` | RRF fusion |
| `reranking.py` | `reranking.rs` | Cross-encoder reranking |

#### 10. Agents (`hf_gtc.agents` → `agents`) - P2

**Python Reference**: 4 files, 171 lines API

| Python File | Rust Equivalent | Key Types |
|-------------|-----------------|-----------|
| `memory.py` | `memory.rs` | Working/long-term memory |
| `planning.py` | `planning.rs` | ReAct, tree search |
| `tools.py` | `tools.rs` | Tool definitions |

#### 11. Safety (`hf_gtc.safety` → `safety`) - P2

**Python Reference**: 4 files, 192 lines API

| Python File | Rust Equivalent | Key Types |
|-------------|-----------------|-----------|
| `guardrails.py` | `guardrails.rs` | Input/output validation |
| `privacy.py` | `privacy.rs` | Differential privacy (Poka-Yoke tiers) |
| `watermarking.py` | `watermarking.rs` | Model watermarking |

#### 12-13. Audio/Multimodal - P3

Lower priority, maps to whisper-apr and future stack components.

---

## Dependencies

### Production Dependencies

```toml
[dependencies]
# Sovereign AI Stack ONLY - zero external dependencies
trueno = "0.14"     # SIMD tensor operations
aprender = "0.25"   # ML algorithms

# Future (P1 modules):
# trueno-rag = "0.1"    # RAG pipeline
# trueno-db = "0.3"     # Vector database
# pacha = "0.2"         # Model registry
```

### Development Dependencies

```toml
[dev-dependencies]
jugar-probar = "1.0"  # Property-based testing (Hypothesis equivalent)
```

---

## Quality Gates

### Coverage Requirements

| Metric | Target | Enforcement | Python Equivalent |
|--------|--------|-------------|-------------------|
| Line Coverage | ≥95% | `make coverage` | pytest --cov-fail-under=95 |
| File Coverage | ≥95% per file | Manual review | Per-file enforcement |
| Mutation Score | ≥80% | `make mutants` | mutmut + Hypothesis |
| Property Tests | 100 examples | jugar-probar | Hypothesis (100 examples) |

### Testing Tiers

| Tier | Target | Rust | Python Equivalent |
|------|--------|------|-------------------|
| 1 (ON-SAVE) | <1s | `cargo check && clippy` | ruff check + ty check |
| 2 (PRE-COMMIT) | <30s | `make tier2` | pytest -x --ff |
| 3 (PRE-PUSH) | <5min | `make quality-gates` | Full test suite |

---

## Implementation Status

### Phase 1: Core Infrastructure ✅ (P0)

- [x] Project scaffold (Cargo.toml, Makefile)
- [x] Error types without external dependencies
- [x] Module structure with 6 modules
- [x] 223 passing tests

### Phase 2: Full P0 Parity 🔄

- [x] `preprocessing` - tokenization, normalization, augmentation
- [x] `models` - tree, boosting, transformer configs
- [x] `training` - trainer, callbacks, schedulers
- [x] `inference` - pipeline, batch processing
- [x] `evaluation` - metrics, cross-validation
- [x] `deployment` - export, quantization
- [ ] Property-based tests with jugar-probar
- [ ] Integration tests with trueno tensors

### Phase 3: P1 Modules (Hub, Generation, RAG)

- [ ] `hub` - model search, versioning, registry
- [ ] `generation` - prompting, chat, sampling
- [ ] `rag` - vectorstore, chunking, retrieval

### Phase 4: P2 Modules (Agents, Safety)

- [ ] `agents` - memory, planning, tools
- [ ] `safety` - guardrails, privacy, watermarking

### Phase 5: P3 Modules (Audio, Multimodal)

- [ ] `audio` - speech, music (maps to whisper-apr)
- [ ] `multimodal` - video, document, vision

---

## API Design Patterns

Following `hf-ground-truth-corpus` patterns for consistency:

### 1. Config Structs (Dataclass Equivalent)

```rust
/// Configuration for [Feature]
///
/// # Python Equivalent (hf_gtc)
/// ```python
/// from hf_gtc.[module] import [Feature]Config
/// config = [Feature]Config(param=value)
/// ```
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Description of param
    pub param: Type,
}
```

### 2. Factory Functions

```rust
/// Create a [feature] with default settings
pub fn create_feature() -> Feature { ... }

/// Create a [feature] from config
pub fn create_feature_from_config(config: FeatureConfig) -> Feature { ... }
```

### 3. Validation Functions

```rust
/// Validate [feature] configuration
pub fn validate_config(config: &FeatureConfig) -> Result<(), CorpusError> { ... }
```

### 4. Calculator Functions

```rust
/// Calculate [metric] for [feature]
pub fn calculate_metric(input: Input) -> Output { ... }
```

### 5. Listing Functions

```rust
/// List supported [feature] types
pub fn list_supported_types() -> Vec<FeatureType> { ... }
```

---

## Usage with Batuta RAG Oracle

This corpus is indexed by Batuta's RAG Oracle for cross-language pattern discovery:

```bash
# Index both corpora
batuta oracle --rag-index

# Query for patterns (returns both Python and Rust)
batuta oracle --rag "how to implement LoRA fine-tuning"
# Returns: hf_gtc/training/lora.py + training/lora.rs (planned)

batuta oracle --rag "batch processing with padding"
# Returns: hf_gtc/inference/batching.py + inference/batch.rs
```

---

## Development Workflow

### Quick Start

```bash
# Clone and build
git clone https://github.com/paiml/batuta-ground-truth-mlops-corpus
cd batuta-ground-truth-mlops-corpus
cargo build

# Run tests (223 passing)
make test

# Run with coverage (95% target)
make coverage

# Run examples
make examples
```

### Adding New Patterns

1. Check `hf-ground-truth-corpus` for Python reference
2. Create Rust module with same API structure
3. Add comprehensive tests (target: 95%+ coverage)
4. Add Python equivalent in docstrings
5. Run `make quality-gates`
6. Update this specification

---

## File Size Limits

| Metric | Limit | Rationale |
|--------|-------|-----------|
| Lines per file | <500 | Maintainability |
| Functions per module | <20 | Single responsibility |
| Test count per file | ≥10 | Coverage compliance |

---

## Changelog

### v0.2.0 (2026-01-31)

- Updated specification for full parity with hf-ground-truth-corpus
- Added detailed module mapping (14 modules)
- Added cross-language examples for all implemented modules
- Added P1/P2/P3 priority roadmap
- Added API design patterns section

### v0.1.0 (2026-01-31)

- Initial scaffold with 6 modules (P0)
- 223 passing tests
- Makefile with realizar-style coverage
- Zero external dependencies (trueno + aprender only)
