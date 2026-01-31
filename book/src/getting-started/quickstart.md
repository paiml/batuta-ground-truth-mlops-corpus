# Quick Start

This guide shows how to use the most common patterns in the corpus.

## Tokenization

```rust
use batuta_ground_truth_mlops_corpus::preprocessing::{
    Tokenizer, TokenizerConfig,
};

let tokenizer = Tokenizer::whitespace();
let tokens = tokenizer.tokenize("Hello, world!");

for token in tokens {
    println!("{}: [{}, {}]", token.text, token.start, token.end);
}
```

## Model Configuration

```rust
use batuta_ground_truth_mlops_corpus::models::{
    RandomForestConfig, TransformerConfig,
};

// Random Forest
let rf = RandomForestConfig::default()
    .n_estimators(200)
    .max_depth(15);

// Transformer
let transformer = TransformerConfig::bert_base();
println!("Hidden size: {}", transformer.hidden_size);
```

## Training Loop

```rust
use batuta_ground_truth_mlops_corpus::training::{
    TrainerConfig, Trainer, TrainingMetrics,
    EarlyStopping, LearningRateScheduler,
};

// Configure trainer
let config = TrainerConfig::default()
    .epochs(20)
    .batch_size(32)
    .learning_rate(0.001)
    .early_stopping(5);

let mut trainer = Trainer::new(config);

// Learning rate scheduler
let mut scheduler = LearningRateScheduler::cosine_annealing(0.001, 100, 0.0);

// Training loop (pseudo-code)
for epoch in 1..=20 {
    // ... train step ...
    scheduler.step();

    let metrics = TrainingMetrics {
        epoch,
        train_loss: 0.5,
        val_loss: Some(0.6),
        ..Default::default()
    };
    trainer.record_epoch(metrics);
}
```

## Text Generation

```rust
use batuta_ground_truth_mlops_corpus::generation::{
    SamplingConfig, SamplingStrategy,
    ChatMessage, Conversation,
};

// Sampling configuration
let config = SamplingConfig::new()
    .strategy(SamplingStrategy::TopP)
    .temperature(0.7)
    .top_p(0.9);

// Chat conversation
let mut conv = Conversation::new();
conv.add_system("You are a helpful assistant.");
conv.add_user("What is Rust?");
conv.add_assistant("Rust is a systems programming language.");
```

## RAG Pipeline

```rust
use batuta_ground_truth_mlops_corpus::rag::{
    ChunkConfig, ChunkingStrategy, chunk_document,
    RagConfig, RetrievalMethod,
};

// Chunking
let chunk_config = ChunkConfig::new()
    .chunk_size(512)
    .overlap(50)
    .strategy(ChunkingStrategy::Semantic);

let document = "Your long document text here...";
let result = chunk_document(document, &chunk_config);
println!("Created {} chunks", result.total_chunks);

// RAG configuration
let rag = RagConfig::new()
    .method(RetrievalMethod::Hybrid)
    .top_k(5);
```

## Next Steps

- Explore the [Examples](../examples/tokenization.md) section for detailed usage
- Check the [API Reference](../api/preprocessing.md) for complete documentation
