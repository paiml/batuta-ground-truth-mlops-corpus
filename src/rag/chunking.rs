//! Document Chunking
//!
//! Split documents into chunks for RAG indexing.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.rag import ChunkingStrategy, create_chunking_config
//! config = create_chunking_config(chunk_size=512, overlap=50)
//! ```

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks (default)
    #[default]
    FixedSize,
    /// Sentence-based chunking
    Sentence,
    /// Paragraph-based chunking
    Paragraph,
    /// Semantic chunking (by similarity)
    Semantic,
    /// Recursive character splitting
    Recursive,
}

impl ChunkingStrategy {
    /// Get strategy name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FixedSize => "fixed_size",
            Self::Sentence => "sentence",
            Self::Paragraph => "paragraph",
            Self::Semantic => "semantic",
            Self::Recursive => "recursive",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fixed_size" | "fixed" | "char" => Some(Self::FixedSize),
            "sentence" | "sent" => Some(Self::Sentence),
            "paragraph" | "para" => Some(Self::Paragraph),
            "semantic" => Some(Self::Semantic),
            "recursive" | "recursive_char" => Some(Self::Recursive),
            _ => None,
        }
    }

    /// List all strategies
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::FixedSize,
            Self::Sentence,
            Self::Paragraph,
            Self::Semantic,
            Self::Recursive,
        ]
    }
}

/// Overlap handling type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlapType {
    /// Overlap by character count (default)
    #[default]
    Characters,
    /// Overlap by token count
    Tokens,
    /// Overlap by percentage
    Percentage,
    /// No overlap
    None,
}

impl OverlapType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Characters => "characters",
            Self::Tokens => "tokens",
            Self::Percentage => "percentage",
            Self::None => "none",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "characters" | "chars" | "char" => Some(Self::Characters),
            "tokens" | "token" => Some(Self::Tokens),
            "percentage" | "percent" | "pct" => Some(Self::Percentage),
            "none" | "no" => Some(Self::None),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![Self::Characters, Self::Tokens, Self::Percentage, Self::None]
    }
}

/// Boundary detection method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryDetection {
    /// No boundary detection (default)
    #[default]
    None,
    /// Detect sentence boundaries
    Sentence,
    /// Detect paragraph boundaries
    Paragraph,
    /// Detect code block boundaries
    CodeBlock,
}

impl BoundaryDetection {
    /// Get detection name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sentence => "sentence",
            Self::Paragraph => "paragraph",
            Self::CodeBlock => "code_block",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "sentence" | "sent" => Some(Self::Sentence),
            "paragraph" | "para" => Some(Self::Paragraph),
            "code_block" | "code" => Some(Self::CodeBlock),
            _ => None,
        }
    }

    /// List all methods
    pub fn list_all() -> Vec<Self> {
        vec![Self::None, Self::Sentence, Self::Paragraph, Self::CodeBlock]
    }
}

/// Chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Chunking strategy
    pub strategy: ChunkingStrategy,
    /// Target chunk size (in characters or tokens)
    pub chunk_size: usize,
    /// Overlap size
    pub overlap: usize,
    /// Overlap type
    pub overlap_type: OverlapType,
    /// Boundary detection
    pub boundary_detection: BoundaryDetection,
    /// Minimum chunk size (don't create smaller chunks)
    pub min_chunk_size: usize,
    /// Strip whitespace from chunks
    pub strip_whitespace: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkingStrategy::FixedSize,
            chunk_size: 512,
            overlap: 50,
            overlap_type: OverlapType::Characters,
            boundary_detection: BoundaryDetection::None,
            min_chunk_size: 100,
            strip_whitespace: true,
        }
    }
}

impl ChunkConfig {
    /// Create new chunk config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set strategy
    pub fn strategy(mut self, strategy: ChunkingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set overlap
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set overlap type
    pub fn overlap_type(mut self, ot: OverlapType) -> Self {
        self.overlap_type = ot;
        self
    }

    /// Set boundary detection
    pub fn boundary_detection(mut self, bd: BoundaryDetection) -> Self {
        self.boundary_detection = bd;
        self
    }

    /// Set min chunk size
    pub fn min_chunk_size(mut self, size: usize) -> Self {
        self.min_chunk_size = size;
        self
    }

    /// Set strip whitespace
    pub fn strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_size == 0 {
            return Err("Chunk size must be > 0".to_string());
        }
        if self.overlap >= self.chunk_size {
            return Err("Overlap must be less than chunk size".to_string());
        }
        if self.min_chunk_size > self.chunk_size {
            return Err("Min chunk size cannot exceed chunk size".to_string());
        }
        Ok(())
    }

    /// Calculate effective overlap
    pub fn effective_overlap(&self) -> usize {
        match self.overlap_type {
            OverlapType::None => 0,
            OverlapType::Characters | OverlapType::Tokens => self.overlap,
            OverlapType::Percentage => (self.chunk_size * self.overlap) / 100,
        }
    }
}

/// A document chunk
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk content
    pub content: String,
    /// Start offset in original document
    pub start_offset: usize,
    /// End offset in original document
    pub end_offset: usize,
    /// Chunk index
    pub index: usize,
    /// Document ID (if known)
    pub document_id: Option<String>,
    /// Metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Chunk {
    /// Create new chunk
    pub fn new(content: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        Self {
            content: content.into(),
            start_offset: start,
            end_offset: end,
            index,
            document_id: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set document ID
    pub fn with_document_id(mut self, id: impl Into<String>) -> Self {
        self.document_id = Some(id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get chunk length
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Result of chunking a document
#[derive(Debug, Clone, Default)]
pub struct ChunkResult {
    /// Generated chunks
    pub chunks: Vec<Chunk>,
    /// Original document length
    pub original_length: usize,
    /// Total chunks created
    pub total_chunks: usize,
    /// Average chunk size
    pub avg_chunk_size: f64,
}

impl ChunkResult {
    /// Create new result
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from chunks
    pub fn from_chunks(chunks: Vec<Chunk>, original_length: usize) -> Self {
        let total_chunks = chunks.len();
        let avg_chunk_size = if total_chunks > 0 {
            chunks.iter().map(|c| c.len()).sum::<usize>() as f64 / total_chunks as f64
        } else {
            0.0
        };

        Self {
            chunks,
            original_length,
            total_chunks,
            avg_chunk_size,
        }
    }
}

/// Chunk a document with the given config
pub fn chunk_document(text: &str, config: &ChunkConfig) -> ChunkResult {
    if text.is_empty() {
        return ChunkResult::new();
    }

    let mut chunks = Vec::new();
    let step = config.chunk_size - config.effective_overlap();
    let step = step.max(1);

    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;
    let mut index = 0;

    while start < chars.len() {
        let end = (start + config.chunk_size).min(chars.len());
        let content: String = chars[start..end].iter().collect();

        let content = if config.strip_whitespace {
            content.trim().to_string()
        } else {
            content
        };

        if content.len() >= config.min_chunk_size || start + step >= chars.len() {
            chunks.push(Chunk::new(content, start, end, index));
            index += 1;
        }

        start += step;
    }

    ChunkResult::from_chunks(chunks, text.len())
}

/// Calculate expected chunk count
pub fn calculate_chunk_count(doc_length: usize, chunk_size: usize, overlap: usize) -> usize {
    if doc_length == 0 || chunk_size == 0 {
        return 0;
    }
    let step = (chunk_size - overlap).max(1);
    doc_length.div_ceil(step)
}

/// Calculate overlap ratio
pub fn calculate_overlap_ratio(overlap: usize, chunk_size: usize) -> f64 {
    if chunk_size == 0 {
        return 0.0;
    }
    overlap as f64 / chunk_size as f64
}

/// Get recommended chunk size for a model
pub fn get_recommended_chunk_size(model_context: usize, safety_margin: f64) -> usize {
    ((model_context as f64) * safety_margin) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_strategy_default() {
        assert_eq!(ChunkingStrategy::default(), ChunkingStrategy::FixedSize);
    }

    #[test]
    fn test_chunking_strategy_as_str() {
        assert_eq!(ChunkingStrategy::FixedSize.as_str(), "fixed_size");
        assert_eq!(ChunkingStrategy::Sentence.as_str(), "sentence");
        assert_eq!(ChunkingStrategy::Semantic.as_str(), "semantic");
    }

    #[test]
    fn test_chunking_strategy_from_str() {
        assert_eq!(ChunkingStrategy::parse("fixed"), Some(ChunkingStrategy::FixedSize));
        assert_eq!(ChunkingStrategy::parse("sent"), Some(ChunkingStrategy::Sentence));
        assert_eq!(ChunkingStrategy::parse("unknown"), None);
    }

    #[test]
    fn test_chunking_strategy_list_all() {
        assert_eq!(ChunkingStrategy::list_all().len(), 5);
    }

    #[test]
    fn test_overlap_type_default() {
        assert_eq!(OverlapType::default(), OverlapType::Characters);
    }

    #[test]
    fn test_overlap_type_as_str() {
        assert_eq!(OverlapType::Characters.as_str(), "characters");
        assert_eq!(OverlapType::Tokens.as_str(), "tokens");
    }

    #[test]
    fn test_overlap_type_from_str() {
        assert_eq!(OverlapType::parse("chars"), Some(OverlapType::Characters));
        assert_eq!(OverlapType::parse("pct"), Some(OverlapType::Percentage));
    }

    #[test]
    fn test_boundary_detection_default() {
        assert_eq!(BoundaryDetection::default(), BoundaryDetection::None);
    }

    #[test]
    fn test_boundary_detection_from_str() {
        assert_eq!(BoundaryDetection::parse("code"), Some(BoundaryDetection::CodeBlock));
    }

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.overlap, 50);
        assert!(config.strip_whitespace);
    }

    #[test]
    fn test_chunk_config_builder() {
        let config = ChunkConfig::new()
            .strategy(ChunkingStrategy::Sentence)
            .chunk_size(1024)
            .overlap(100)
            .overlap_type(OverlapType::Tokens)
            .boundary_detection(BoundaryDetection::Sentence)
            .min_chunk_size(50)
            .strip_whitespace(false);

        assert_eq!(config.strategy, ChunkingStrategy::Sentence);
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.overlap, 100);
        assert_eq!(config.overlap_type, OverlapType::Tokens);
        assert_eq!(config.boundary_detection, BoundaryDetection::Sentence);
        assert_eq!(config.min_chunk_size, 50);
        assert!(!config.strip_whitespace);
    }

    #[test]
    fn test_chunk_config_validate() {
        let valid = ChunkConfig::default();
        assert!(valid.validate().is_ok());

        let zero_size = ChunkConfig::new().chunk_size(0);
        assert!(zero_size.validate().is_err());

        let big_overlap = ChunkConfig::new().chunk_size(100).overlap(200);
        assert!(big_overlap.validate().is_err());

        let big_min = ChunkConfig::new().chunk_size(100).min_chunk_size(200);
        assert!(big_min.validate().is_err());
    }

    #[test]
    fn test_chunk_config_effective_overlap() {
        let chars = ChunkConfig::new().overlap(50);
        assert_eq!(chars.effective_overlap(), 50);

        let none = ChunkConfig::new().overlap_type(OverlapType::None).overlap(100);
        assert_eq!(none.effective_overlap(), 0);

        let pct = ChunkConfig::new()
            .chunk_size(1000)
            .overlap(10)
            .overlap_type(OverlapType::Percentage);
        assert_eq!(pct.effective_overlap(), 100);
    }

    #[test]
    fn test_chunk_new() {
        let chunk = Chunk::new("Hello world", 0, 11, 0);
        assert_eq!(chunk.content, "Hello world");
        assert_eq!(chunk.start_offset, 0);
        assert_eq!(chunk.end_offset, 11);
        assert_eq!(chunk.index, 0);
    }

    #[test]
    fn test_chunk_with_document_id() {
        let chunk = Chunk::new("test", 0, 4, 0).with_document_id("doc1");
        assert_eq!(chunk.document_id, Some("doc1".to_string()));
    }

    #[test]
    fn test_chunk_with_metadata() {
        let chunk = Chunk::new("test", 0, 4, 0).with_metadata("source", "file.txt");
        assert_eq!(chunk.metadata.get("source"), Some(&"file.txt".to_string()));
    }

    #[test]
    fn test_chunk_len() {
        let chunk = Chunk::new("Hello", 0, 5, 0);
        assert_eq!(chunk.len(), 5);
        assert!(!chunk.is_empty());

        let empty = Chunk::new("", 0, 0, 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_chunk_result_new() {
        let result = ChunkResult::new();
        assert!(result.chunks.is_empty());
        assert_eq!(result.total_chunks, 0);
    }

    #[test]
    fn test_chunk_result_from_chunks() {
        let chunks = vec![
            Chunk::new("aaaa", 0, 4, 0),
            Chunk::new("bbbb", 4, 8, 1),
        ];
        let result = ChunkResult::from_chunks(chunks, 8);
        
        assert_eq!(result.total_chunks, 2);
        assert_eq!(result.original_length, 8);
        assert_eq!(result.avg_chunk_size, 4.0);
    }

    #[test]
    fn test_chunk_document_empty() {
        let config = ChunkConfig::default();
        let result = chunk_document("", &config);
        assert!(result.chunks.is_empty());
    }

    #[test]
    fn test_chunk_document_small() {
        let config = ChunkConfig::new().chunk_size(100).min_chunk_size(0);
        let result = chunk_document("Hello world", &config);
        assert_eq!(result.total_chunks, 1);
        assert_eq!(result.chunks[0].content, "Hello world");
    }

    #[test]
    fn test_chunk_document_with_overlap() {
        let config = ChunkConfig::new()
            .chunk_size(10)
            .overlap(5)
            .min_chunk_size(1);
        let text = "0123456789abcdefghij";
        let result = chunk_document(text, &config);
        
        assert!(result.total_chunks > 1);
        // Verify overlap: second chunk should start where first ends minus overlap
    }

    #[test]
    fn test_calculate_chunk_count() {
        assert_eq!(calculate_chunk_count(0, 100, 0), 0);
        assert_eq!(calculate_chunk_count(100, 0, 0), 0);
        assert_eq!(calculate_chunk_count(100, 100, 0), 1);
        assert_eq!(calculate_chunk_count(200, 100, 0), 2);
        assert_eq!(calculate_chunk_count(150, 100, 50), 3);
    }

    #[test]
    fn test_calculate_overlap_ratio() {
        assert_eq!(calculate_overlap_ratio(50, 100), 0.5);
        assert_eq!(calculate_overlap_ratio(0, 100), 0.0);
        assert_eq!(calculate_overlap_ratio(50, 0), 0.0);
    }

    #[test]
    fn test_get_recommended_chunk_size() {
        let size = get_recommended_chunk_size(4096, 0.5);
        assert_eq!(size, 2048);

        let size = get_recommended_chunk_size(8192, 0.25);
        assert_eq!(size, 2048);
    }
}
