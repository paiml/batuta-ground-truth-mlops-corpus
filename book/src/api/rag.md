# RAG API

## Module: `batuta_ground_truth_mlops_corpus::rag`

### ChunkConfig

```rust
pub struct ChunkConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub strategy: ChunkingStrategy,
}

impl ChunkConfig {
    pub fn new() -> Self;
    pub fn chunk_size(self, size: usize) -> Self;
    pub fn overlap(self, overlap: usize) -> Self;
    pub fn strategy(self, s: ChunkingStrategy) -> Self;
}
```

### ChunkingStrategy

```rust
pub enum ChunkingStrategy {
    Fixed,
    Sentence,
    Paragraph,
    Semantic,
}
```

### chunk_document

```rust
pub fn chunk_document(text: &str, config: &ChunkConfig) -> ChunkResult;

pub struct ChunkResult {
    pub chunks: Vec<Chunk>,
    pub total_chunks: usize,
}
```

### RagConfig

```rust
pub struct RagConfig {
    pub method: RetrievalMethod,
    pub distance_metric: DistanceMetric,
    pub top_k: usize,
}

impl RagConfig {
    pub fn new() -> Self;
    pub fn method(self, m: RetrievalMethod) -> Self;
    pub fn distance_metric(self, d: DistanceMetric) -> Self;
    pub fn top_k(self, k: usize) -> Self;
}
```

### Scoring Functions

```rust
pub fn calculate_rrf_score(rank: usize, k: usize) -> f64;

pub fn calculate_bm25_score(
    tf: f64,
    df: f64,
    num_docs: f64,
    doc_len: f64,
    config: &Bm25Config,
) -> f64;
```
