# RAG Pipeline Example

Run with: `cargo run --example rag_demo`

## Overview

Demonstrates Retrieval-Augmented Generation (RAG) pipeline components:
- Document chunking
- Retrieval configuration
- Reranking
- Scoring functions (BM25, RRF)

## Code Highlights

```rust
use batuta_ground_truth_mlops_corpus::rag::{
    ChunkConfig, ChunkingStrategy, chunk_document,
    RagConfig, RetrievalMethod, DistanceMetric,
    RerankerConfig, RerankerType,
    Bm25Config, calculate_rrf_score, calculate_bm25_score,
};

fn main() {
    // Chunking configuration
    let chunk_config = ChunkConfig::new()
        .chunk_size(512)
        .overlap(50)
        .strategy(ChunkingStrategy::Semantic);

    // Chunk a document
    let document = "Your long document text...";
    let result = chunk_document(document, &chunk_config);
    println!("Created {} chunks", result.total_chunks);

    // RAG configuration
    let rag = RagConfig::new()
        .method(RetrievalMethod::Hybrid)
        .distance_metric(DistanceMetric::Cosine)
        .top_k(5);

    // Reranker (cross-encoder)
    let reranker = RerankerConfig::new()
        .reranker_type(RerankerType::CrossEncoder)
        .model("cross-encoder/ms-marco")
        .top_n(3);

    // RRF scoring
    let rrf_score = calculate_rrf_score(1, 60);
    println!("RRF score for rank 1: {:.4}", rrf_score);

    // BM25 scoring
    let bm25 = Bm25Config::default();
    let score = calculate_bm25_score(3.0, 50.0, 1000.0, 100.0, &bm25);
}
```

## Chunking Strategies

- `Fixed` - Fixed-size chunks
- `Sentence` - Sentence-based chunking
- `Paragraph` - Paragraph-based chunking
- `Semantic` - Semantic similarity-based chunking

## Retrieval Methods

- `Dense` - Vector similarity search
- `Sparse` - BM25/keyword search
- `Hybrid` - Combined dense + sparse

## Fusion Methods

- `RRF` - Reciprocal Rank Fusion
- `Linear` - Linear combination
- `CombSum` - Combined sum
- `CombMnz` - Combined MNZ
