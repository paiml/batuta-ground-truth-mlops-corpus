//! RAG (Retrieval-Augmented Generation) demo

use batuta_ground_truth_mlops_corpus::rag::{
    ChunkConfig, ChunkingStrategy, chunk_document,
    RagConfig, RetrievalMethod, DistanceMetric,
    RerankerConfig, RerankerType, FusionMethod,
    Bm25Config, calculate_rrf_score, calculate_bm25_score,
};

fn main() {
    println!("=== RAG Pipeline Demo ===\n");

    // Text chunking configuration
    println!("Chunking Configuration:");
    let chunk_config = ChunkConfig::new()
        .chunk_size(512)
        .overlap(50)
        .strategy(ChunkingStrategy::Semantic);

    println!("  chunk_size: {}", chunk_config.chunk_size);
    println!("  overlap: {}", chunk_config.overlap);
    println!("  strategy: {:?}", chunk_config.strategy);

    // Chunk a sample document
    let document = r#"
# Introduction to Rust

Rust is a systems programming language focused on safety, concurrency, and performance.

## Memory Safety

Rust achieves memory safety without garbage collection through its ownership system.
The compiler checks ownership rules at compile time.

## Concurrency

Rust's type system prevents data races at compile time.
This makes concurrent programming much safer.

## Performance

Rust has no runtime or garbage collector, giving predictable performance.
It's suitable for embedded systems and high-performance applications.
"#;

    let result = chunk_document(document, &chunk_config);

    println!("\nChunked Document ({} chunks):", result.total_chunks);
    for (i, chunk) in result.chunks.iter().enumerate() {
        let preview: String = chunk.content.chars().take(60).collect();
        println!("  Chunk {}: \"{}...\" ({} chars)", i + 1, preview.trim(), chunk.content.len());
    }

    // RAG configuration
    println!("\n--- RAG Configuration ---");
    let rag_config = RagConfig::new()
        .method(RetrievalMethod::Hybrid)
        .distance_metric(DistanceMetric::Cosine)
        .top_k(5);

    println!("  method: {:?}", rag_config.method);
    println!("  distance_metric: {:?}", rag_config.distance_metric);
    println!("  top_k: {}", rag_config.top_k);

    // Reranker configuration
    println!("\n--- Reranker Configuration ---");
    let cross_encoder = RerankerConfig::new()
        .reranker_type(RerankerType::CrossEncoder)
        .model("cross-encoder/ms-marco")
        .top_n(3);
    println!("Cross-Encoder Reranker:");
    println!("  type: {:?}", cross_encoder.reranker_type);
    println!("  top_n: {}", cross_encoder.top_n);

    // Scoring functions
    println!("\n--- Scoring Functions ---");

    // RRF (Reciprocal Rank Fusion) scores
    println!("\nRRF Scores (k=60):");
    for rank in 1..=5 {
        let score = calculate_rrf_score(rank, 60);
        println!("  Rank {}: {:.4}", rank, score);
    }

    // BM25 score example
    let bm25_config = Bm25Config::default();
    let bm25_score = calculate_bm25_score(
        3.0,   // term frequency
        50.0,  // document frequency
        1000.0, // total documents
        100.0, // document length
        &bm25_config,
    );
    println!("\nBM25 Score Example: {:.4}", bm25_score);

    // Fusion methods
    println!("\n--- Fusion Methods ---");
    let methods = [
        FusionMethod::RRF,
        FusionMethod::Linear,
        FusionMethod::CombSum,
        FusionMethod::CombMnz,
    ];
    for method in &methods {
        println!("  - {:?}", method);
    }
}
