//! Document Processing
//!
//! Document understanding and layout analysis.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.multimodal import DocumentType, DocumentConfig, process_document
//! config = DocumentConfig(ocr=True, layout=True)
//! ```

/// Document type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DocumentType {
    /// PDF document (default)
    #[default]
    Pdf,
    /// Scanned image
    Image,
    /// Office document (Word, etc.)
    Office,
    /// Webpage/HTML
    Html,
    /// Plain text
    Text,
    /// Spreadsheet
    Spreadsheet,
}

impl DocumentType {
    /// Get type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Image => "image",
            Self::Office => "office",
            Self::Html => "html",
            Self::Text => "text",
            Self::Spreadsheet => "spreadsheet",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pdf" => Some(Self::Pdf),
            "image" | "img" | "scan" => Some(Self::Image),
            "office" | "doc" | "docx" | "word" => Some(Self::Office),
            "html" | "web" | "webpage" => Some(Self::Html),
            "text" | "txt" | "plain" => Some(Self::Text),
            "spreadsheet" | "xlsx" | "excel" | "csv" => Some(Self::Spreadsheet),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::Pdf,
            Self::Image,
            Self::Office,
            Self::Html,
            Self::Text,
            Self::Spreadsheet,
        ]
    }

    /// Check if OCR may be needed
    pub fn may_need_ocr(&self) -> bool {
        matches!(self, Self::Pdf | Self::Image)
    }
}

/// Layout element type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutElement {
    /// Text paragraph (default)
    #[default]
    Paragraph,
    /// Heading/title
    Heading,
    /// List item
    ListItem,
    /// Table
    Table,
    /// Figure/image
    Figure,
    /// Caption
    Caption,
    /// Footer/header
    Footer,
    /// Page number
    PageNumber,
}

impl LayoutElement {
    /// Get element name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Paragraph => "paragraph",
            Self::Heading => "heading",
            Self::ListItem => "list_item",
            Self::Table => "table",
            Self::Figure => "figure",
            Self::Caption => "caption",
            Self::Footer => "footer",
            Self::PageNumber => "page_number",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "paragraph" | "text" | "p" => Some(Self::Paragraph),
            "heading" | "title" | "h" => Some(Self::Heading),
            "list_item" | "list" | "li" => Some(Self::ListItem),
            "table" => Some(Self::Table),
            "figure" | "image" | "img" => Some(Self::Figure),
            "caption" => Some(Self::Caption),
            "footer" | "header" => Some(Self::Footer),
            "page_number" | "page" => Some(Self::PageNumber),
            _ => None,
        }
    }

    /// Check if element contains text
    pub fn has_text(&self) -> bool {
        matches!(
            self,
            Self::Paragraph | Self::Heading | Self::ListItem | Self::Caption | Self::Footer
        )
    }
}

/// OCR engine type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrEngine {
    /// Tesseract (default)
    #[default]
    Tesseract,
    /// EasyOCR
    EasyOcr,
    /// PaddleOCR
    PaddleOcr,
    /// TrOCR (Transformer)
    TrOcr,
    /// None (text extraction only)
    None,
}

impl OcrEngine {
    /// Get engine name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tesseract => "tesseract",
            Self::EasyOcr => "easyocr",
            Self::PaddleOcr => "paddleocr",
            Self::TrOcr => "trocr",
            Self::None => "none",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tesseract" => Some(Self::Tesseract),
            "easyocr" | "easy" => Some(Self::EasyOcr),
            "paddleocr" | "paddle" => Some(Self::PaddleOcr),
            "trocr" | "transformer" => Some(Self::TrOcr),
            "none" | "disabled" => Some(Self::None),
            _ => None,
        }
    }

    /// Check if GPU accelerated
    pub fn gpu_accelerated(&self) -> bool {
        matches!(self, Self::EasyOcr | Self::PaddleOcr | Self::TrOcr)
    }
}

/// Document processing configuration
#[derive(Debug, Clone)]
pub struct DocumentConfig {
    /// Document type
    pub doc_type: DocumentType,
    /// Enable OCR
    pub ocr_enabled: bool,
    /// OCR engine
    pub ocr_engine: OcrEngine,
    /// Enable layout analysis
    pub layout_analysis: bool,
    /// OCR language
    pub language: String,
    /// Max pages to process (0 = all)
    pub max_pages: usize,
    /// DPI for rendering
    pub dpi: u32,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            doc_type: DocumentType::Pdf,
            ocr_enabled: true,
            ocr_engine: OcrEngine::Tesseract,
            layout_analysis: true,
            language: "eng".to_string(),
            max_pages: 0,
            dpi: 300,
        }
    }
}

impl DocumentConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set document type
    pub fn doc_type(mut self, t: DocumentType) -> Self {
        self.doc_type = t;
        self
    }

    /// Enable/disable OCR
    pub fn ocr_enabled(mut self, enabled: bool) -> Self {
        self.ocr_enabled = enabled;
        self
    }

    /// Set OCR engine
    pub fn ocr_engine(mut self, engine: OcrEngine) -> Self {
        self.ocr_engine = engine;
        self
    }

    /// Enable/disable layout analysis
    pub fn layout_analysis(mut self, enabled: bool) -> Self {
        self.layout_analysis = enabled;
        self
    }

    /// Set OCR language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Set max pages
    pub fn max_pages(mut self, n: usize) -> Self {
        self.max_pages = n;
        self
    }

    /// Set DPI
    pub fn dpi(mut self, d: u32) -> Self {
        self.dpi = d;
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.dpi == 0 {
            return Err("DPI must be > 0".to_string());
        }
        if self.language.is_empty() && self.ocr_enabled {
            return Err("Language required when OCR is enabled".to_string());
        }
        Ok(())
    }
}

/// Bounding box for layout elements
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundingBox {
    /// X coordinate (left)
    pub x: f32,
    /// Y coordinate (top)
    pub y: f32,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl BoundingBox {
    /// Create new bounding box
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Get area
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Get center point
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Check if contains point
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }

    /// Calculate IoU with another box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x_overlap = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let y_overlap = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);

        if x_overlap <= 0.0 || y_overlap <= 0.0 {
            return 0.0;
        }

        let intersection = x_overlap * y_overlap;
        let union = self.area() + other.area() - intersection;

        if union == 0.0 {
            return 0.0;
        }

        intersection / union
    }
}

/// Document region with content
#[derive(Debug, Clone)]
pub struct DocumentRegion {
    /// Element type
    pub element_type: LayoutElement,
    /// Bounding box
    pub bbox: BoundingBox,
    /// Text content
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Page number (0-indexed)
    pub page: usize,
}

impl DocumentRegion {
    /// Create new region
    pub fn new(element_type: LayoutElement, bbox: BoundingBox) -> Self {
        Self {
            element_type,
            bbox,
            text: String::new(),
            confidence: 1.0,
            page: 0,
        }
    }

    /// Set text
    pub fn text(mut self, t: impl Into<String>) -> Self {
        self.text = t.into();
        self
    }

    /// Set confidence
    pub fn confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Set page
    pub fn page(mut self, p: usize) -> Self {
        self.page = p;
        self
    }

    /// Get word count
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Document processing result
#[derive(Debug, Clone)]
pub struct DocumentResult {
    /// Document type
    pub doc_type: DocumentType,
    /// Full extracted text
    pub text: String,
    /// Detected regions
    pub regions: Vec<DocumentRegion>,
    /// Number of pages
    pub num_pages: usize,
    /// Processing time (ms)
    pub processing_time_ms: u64,
}

impl DocumentResult {
    /// Create new result
    pub fn new(doc_type: DocumentType) -> Self {
        Self {
            doc_type,
            text: String::new(),
            regions: Vec::new(),
            num_pages: 0,
            processing_time_ms: 0,
        }
    }

    /// Set text
    pub fn text(mut self, t: impl Into<String>) -> Self {
        self.text = t.into();
        self
    }

    /// Add region
    pub fn add_region(mut self, region: DocumentRegion) -> Self {
        self.regions.push(region);
        self
    }

    /// Set num pages
    pub fn num_pages(mut self, n: usize) -> Self {
        self.num_pages = n;
        self
    }

    /// Set processing time
    pub fn processing_time(mut self, ms: u64) -> Self {
        self.processing_time_ms = ms;
        self
    }

    /// Get word count
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Get regions by type
    pub fn regions_of_type(&self, element_type: LayoutElement) -> Vec<&DocumentRegion> {
        self.regions
            .iter()
            .filter(|r| r.element_type == element_type)
            .collect()
    }

    /// Get average confidence
    pub fn average_confidence(&self) -> f32 {
        if self.regions.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.regions.iter().map(|r| r.confidence).sum();
        sum / self.regions.len() as f32
    }
}

/// Document processing statistics
#[derive(Debug, Clone, Default)]
pub struct DocumentStats {
    /// Documents processed
    pub documents_processed: usize,
    /// Total pages
    pub total_pages: usize,
    /// Total words extracted
    pub total_words: usize,
    /// Total processing time (ms)
    pub total_time_ms: u64,
}

impl DocumentStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record document processing
    pub fn record(&mut self, result: &DocumentResult) {
        self.documents_processed += 1;
        self.total_pages += result.num_pages;
        self.total_words += result.word_count();
        self.total_time_ms += result.processing_time_ms;
    }

    /// Get average pages per document
    pub fn avg_pages(&self) -> f64 {
        if self.documents_processed == 0 {
            return 0.0;
        }
        self.total_pages as f64 / self.documents_processed as f64
    }

    /// Get pages per second
    pub fn pages_per_second(&self) -> f64 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        self.total_pages as f64 / (self.total_time_ms as f64 / 1000.0)
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Documents: {} docs, {} pages, {} words, {:.1} pages/s",
            self.documents_processed,
            self.total_pages,
            self.total_words,
            self.pages_per_second()
        )
    }
}

/// Estimate processing time for document
pub fn estimate_document_time(config: &DocumentConfig, num_pages: usize) -> u64 {
    let base_time = 100; // Base time per page in ms
    let ocr_factor = if config.ocr_enabled { 5 } else { 1 };
    let layout_factor = if config.layout_analysis { 2 } else { 1 };
    let dpi_factor = config.dpi / 150;

    (base_time * num_pages * ocr_factor * layout_factor * dpi_factor as usize) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_type_default() {
        assert_eq!(DocumentType::default(), DocumentType::Pdf);
    }

    #[test]
    fn test_document_type_as_str() {
        assert_eq!(DocumentType::Pdf.as_str(), "pdf");
        assert_eq!(DocumentType::Image.as_str(), "image");
        assert_eq!(DocumentType::Office.as_str(), "office");
        assert_eq!(DocumentType::Html.as_str(), "html");
        assert_eq!(DocumentType::Text.as_str(), "text");
        assert_eq!(DocumentType::Spreadsheet.as_str(), "spreadsheet");
    }

    #[test]
    fn test_document_type_parse() {
        assert_eq!(DocumentType::parse("pdf"), Some(DocumentType::Pdf));
        assert_eq!(DocumentType::parse("scan"), Some(DocumentType::Image));
        assert_eq!(DocumentType::parse("docx"), Some(DocumentType::Office));
        assert_eq!(DocumentType::parse("webpage"), Some(DocumentType::Html));
        assert_eq!(DocumentType::parse("plain"), Some(DocumentType::Text));
        assert_eq!(DocumentType::parse("excel"), Some(DocumentType::Spreadsheet));
        assert_eq!(DocumentType::parse("unknown"), None);
    }

    #[test]
    fn test_document_type_list_all() {
        assert_eq!(DocumentType::list_all().len(), 6);
    }

    #[test]
    fn test_document_type_may_need_ocr() {
        assert!(DocumentType::Pdf.may_need_ocr());
        assert!(DocumentType::Image.may_need_ocr());
        assert!(!DocumentType::Text.may_need_ocr());
        assert!(!DocumentType::Html.may_need_ocr());
    }

    #[test]
    fn test_layout_element_as_str() {
        assert_eq!(LayoutElement::Paragraph.as_str(), "paragraph");
        assert_eq!(LayoutElement::Heading.as_str(), "heading");
        assert_eq!(LayoutElement::Table.as_str(), "table");
        assert_eq!(LayoutElement::Figure.as_str(), "figure");
    }

    #[test]
    fn test_layout_element_parse() {
        assert_eq!(LayoutElement::parse("text"), Some(LayoutElement::Paragraph));
        assert_eq!(LayoutElement::parse("title"), Some(LayoutElement::Heading));
        assert_eq!(LayoutElement::parse("li"), Some(LayoutElement::ListItem));
        assert_eq!(LayoutElement::parse("table"), Some(LayoutElement::Table));
        assert_eq!(LayoutElement::parse("img"), Some(LayoutElement::Figure));
        assert_eq!(LayoutElement::parse("unknown"), None);
    }

    #[test]
    fn test_layout_element_has_text() {
        assert!(LayoutElement::Paragraph.has_text());
        assert!(LayoutElement::Heading.has_text());
        assert!(!LayoutElement::Figure.has_text());
        assert!(!LayoutElement::Table.has_text());
    }

    #[test]
    fn test_ocr_engine_as_str() {
        assert_eq!(OcrEngine::Tesseract.as_str(), "tesseract");
        assert_eq!(OcrEngine::EasyOcr.as_str(), "easyocr");
        assert_eq!(OcrEngine::PaddleOcr.as_str(), "paddleocr");
        assert_eq!(OcrEngine::TrOcr.as_str(), "trocr");
        assert_eq!(OcrEngine::None.as_str(), "none");
    }

    #[test]
    fn test_ocr_engine_parse() {
        assert_eq!(OcrEngine::parse("tesseract"), Some(OcrEngine::Tesseract));
        assert_eq!(OcrEngine::parse("easy"), Some(OcrEngine::EasyOcr));
        assert_eq!(OcrEngine::parse("paddle"), Some(OcrEngine::PaddleOcr));
        assert_eq!(OcrEngine::parse("transformer"), Some(OcrEngine::TrOcr));
        assert_eq!(OcrEngine::parse("disabled"), Some(OcrEngine::None));
        assert_eq!(OcrEngine::parse("unknown"), None);
    }

    #[test]
    fn test_ocr_engine_gpu_accelerated() {
        assert!(!OcrEngine::Tesseract.gpu_accelerated());
        assert!(OcrEngine::EasyOcr.gpu_accelerated());
        assert!(OcrEngine::TrOcr.gpu_accelerated());
    }

    #[test]
    fn test_document_config_default() {
        let config = DocumentConfig::default();
        assert_eq!(config.doc_type, DocumentType::Pdf);
        assert!(config.ocr_enabled);
        assert!(config.layout_analysis);
        assert_eq!(config.dpi, 300);
    }

    #[test]
    fn test_document_config_builder() {
        let config = DocumentConfig::new()
            .doc_type(DocumentType::Image)
            .ocr_enabled(true)
            .ocr_engine(OcrEngine::EasyOcr)
            .layout_analysis(false)
            .language("deu")
            .max_pages(10)
            .dpi(150);

        assert_eq!(config.doc_type, DocumentType::Image);
        assert_eq!(config.ocr_engine, OcrEngine::EasyOcr);
        assert!(!config.layout_analysis);
        assert_eq!(config.language, "deu");
        assert_eq!(config.max_pages, 10);
        assert_eq!(config.dpi, 150);
    }

    #[test]
    fn test_document_config_validate() {
        let valid = DocumentConfig::default();
        assert!(valid.validate().is_ok());

        let zero_dpi = DocumentConfig::new().dpi(0);
        assert!(zero_dpi.validate().is_err());

        let no_lang = DocumentConfig::new().language("").ocr_enabled(true);
        assert!(no_lang.validate().is_err());
    }

    #[test]
    fn test_bounding_box_new() {
        let bbox = BoundingBox::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 50.0);
    }

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 50.0);
        assert_eq!(bbox.area(), 5000.0);
    }

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        assert_eq!(bbox.center(), (50.0, 50.0));
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        assert!(bbox.contains(50.0, 50.0));
        assert!(bbox.contains(0.0, 0.0));
        assert!(bbox.contains(100.0, 100.0));
        assert!(!bbox.contains(101.0, 50.0));
        assert!(!bbox.contains(-1.0, 50.0));
    }

    #[test]
    fn test_bounding_box_iou() {
        let box1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let box2 = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
        let iou = box1.iou(&box2);

        // Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        assert!((iou - 2500.0 / 17500.0).abs() < 0.001);
    }

    #[test]
    fn test_bounding_box_iou_no_overlap() {
        let box1 = BoundingBox::new(0.0, 0.0, 50.0, 50.0);
        let box2 = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
        assert_eq!(box1.iou(&box2), 0.0);
    }

    #[test]
    fn test_document_region_new() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 50.0);
        let region = DocumentRegion::new(LayoutElement::Paragraph, bbox);

        assert_eq!(region.element_type, LayoutElement::Paragraph);
        assert!(region.text.is_empty());
    }

    #[test]
    fn test_document_region_builder() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 50.0);
        let region = DocumentRegion::new(LayoutElement::Heading, bbox)
            .text("Hello World")
            .confidence(0.95)
            .page(2);

        assert_eq!(region.text, "Hello World");
        assert!((region.confidence - 0.95).abs() < 0.001);
        assert_eq!(region.page, 2);
    }

    #[test]
    fn test_document_region_word_count() {
        let bbox = BoundingBox::default();
        let region = DocumentRegion::new(LayoutElement::Paragraph, bbox)
            .text("Hello world test");

        assert_eq!(region.word_count(), 3);
    }

    #[test]
    fn test_document_result_new() {
        let result = DocumentResult::new(DocumentType::Pdf);
        assert_eq!(result.doc_type, DocumentType::Pdf);
        assert!(result.regions.is_empty());
    }

    #[test]
    fn test_document_result_builder() {
        let bbox = BoundingBox::default();
        let result = DocumentResult::new(DocumentType::Pdf)
            .text("Full document text")
            .add_region(DocumentRegion::new(LayoutElement::Paragraph, bbox))
            .num_pages(5)
            .processing_time(1500);

        assert_eq!(result.text, "Full document text");
        assert_eq!(result.regions.len(), 1);
        assert_eq!(result.num_pages, 5);
        assert_eq!(result.processing_time_ms, 1500);
    }

    #[test]
    fn test_document_result_word_count() {
        let result = DocumentResult::new(DocumentType::Pdf)
            .text("Hello world test example");
        assert_eq!(result.word_count(), 4);
    }

    #[test]
    fn test_document_result_regions_of_type() {
        let bbox = BoundingBox::default();
        let result = DocumentResult::new(DocumentType::Pdf)
            .add_region(DocumentRegion::new(LayoutElement::Paragraph, bbox))
            .add_region(DocumentRegion::new(LayoutElement::Heading, bbox))
            .add_region(DocumentRegion::new(LayoutElement::Paragraph, bbox));

        let paragraphs = result.regions_of_type(LayoutElement::Paragraph);
        assert_eq!(paragraphs.len(), 2);

        let headings = result.regions_of_type(LayoutElement::Heading);
        assert_eq!(headings.len(), 1);
    }

    #[test]
    fn test_document_result_average_confidence() {
        let bbox = BoundingBox::default();
        let result = DocumentResult::new(DocumentType::Pdf)
            .add_region(DocumentRegion::new(LayoutElement::Paragraph, bbox).confidence(0.9))
            .add_region(DocumentRegion::new(LayoutElement::Paragraph, bbox).confidence(0.8));

        assert!((result.average_confidence() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_document_stats_new() {
        let stats = DocumentStats::new();
        assert_eq!(stats.documents_processed, 0);
    }

    #[test]
    fn test_document_stats_record() {
        let mut stats = DocumentStats::new();
        let result = DocumentResult::new(DocumentType::Pdf)
            .text("Hello world")
            .num_pages(3)
            .processing_time(500);

        stats.record(&result);
        stats.record(&result);

        assert_eq!(stats.documents_processed, 2);
        assert_eq!(stats.total_pages, 6);
        assert_eq!(stats.total_words, 4);
        assert_eq!(stats.total_time_ms, 1000);
    }

    #[test]
    fn test_document_stats_avg_pages() {
        let mut stats = DocumentStats::new();
        let result1 = DocumentResult::new(DocumentType::Pdf).num_pages(4);
        let result2 = DocumentResult::new(DocumentType::Pdf).num_pages(6);

        stats.record(&result1);
        stats.record(&result2);

        assert!((stats.avg_pages() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_document_stats_pages_per_second() {
        let mut stats = DocumentStats::new();
        let result = DocumentResult::new(DocumentType::Pdf)
            .num_pages(10)
            .processing_time(1000); // 1 second

        stats.record(&result);

        assert!((stats.pages_per_second() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_document_stats_format() {
        let mut stats = DocumentStats::new();
        let result = DocumentResult::new(DocumentType::Pdf)
            .text("Hello")
            .num_pages(2)
            .processing_time(100);

        stats.record(&result);

        let formatted = stats.format();
        assert!(formatted.contains("1 docs"));
        assert!(formatted.contains("2 pages"));
    }

    #[test]
    fn test_estimate_document_time() {
        let config = DocumentConfig::default();
        let time = estimate_document_time(&config, 10);
        assert!(time > 0);

        let no_ocr_config = DocumentConfig::new().ocr_enabled(false);
        let no_ocr_time = estimate_document_time(&no_ocr_config, 10);

        assert!(time > no_ocr_time); // OCR takes longer
    }
}
