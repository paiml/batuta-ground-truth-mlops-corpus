//! Multimodal Module
//!
//! Vision, document, and multimodal processing.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.multimodal import (
//!     VisionConfig, ImageFormat, process_image,
//!     DocumentConfig, DocumentType, process_document,
//! )
//! ```
//!
//! # Submodules
//! - `vision`: Image processing and visual features
//! - `document`: Document understanding and OCR

pub mod document;
pub mod vision;

// Re-export key types
pub use document::{
    BoundingBox, DocumentConfig, DocumentRegion, DocumentResult, DocumentStats, DocumentType,
    LayoutElement, OcrEngine,
};
pub use vision::{
    ImageFeatures, ImageFormat, Interpolation, VisionConfig, VisionModel, VisionStats,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_re_exports() {
        let _ = ImageFormat::default();
        let _ = VisionConfig::default();
        let _ = VisionModel::default();
        let _ = Interpolation::default();
    }

    #[test]
    fn test_document_re_exports() {
        let _ = DocumentType::default();
        let _ = DocumentConfig::default();
        let _ = LayoutElement::default();
        let _ = OcrEngine::default();
    }

    #[test]
    fn test_integration_vision_pipeline() {
        let config = VisionConfig::clip()
            .size(224)
            .normalize(true);

        let features = ImageFeatures::new(config.model.embed_dim())
            .num_patches(vision::calculate_num_patches(config.size, 16))
            .original_size(640, 480)
            .processing_time(25);

        assert_eq!(config.model, VisionModel::Clip);
        assert_eq!(features.feature_dim, 512);
        assert_eq!(features.num_patches, 196);
    }

    #[test]
    fn test_integration_document_pipeline() {
        let config = DocumentConfig::new()
            .doc_type(DocumentType::Pdf)
            .ocr_enabled(true)
            .ocr_engine(OcrEngine::Tesseract)
            .layout_analysis(true);

        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 50.0);
        let result = DocumentResult::new(DocumentType::Pdf)
            .text("Sample document text")
            .add_region(DocumentRegion::new(LayoutElement::Paragraph, bbox).text("Text"))
            .num_pages(5)
            .processing_time(1000);

        assert!(config.ocr_enabled);
        assert_eq!(result.num_pages, 5);
        assert_eq!(result.regions.len(), 1);
    }

    #[test]
    fn test_integration_vision_document() {
        // Image-based document processing
        let vision_config = VisionConfig::imagenet();
        let doc_config = DocumentConfig::new()
            .doc_type(DocumentType::Image)
            .ocr_enabled(true);

        // Process image for features
        let _image_features = ImageFeatures::new(vision_config.model.embed_dim());

        // Also extract text via OCR
        let doc_result = DocumentResult::new(DocumentType::Image)
            .text("OCR extracted text")
            .num_pages(1);

        assert!(doc_config.doc_type.may_need_ocr());
        assert_eq!(doc_result.word_count(), 3);
    }

    #[test]
    fn test_bounding_box_operations() {
        let box1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let box2 = BoundingBox::new(50.0, 0.0, 100.0, 100.0);

        // Check overlap
        let iou = box1.iou(&box2);
        assert!(iou > 0.0);

        // Check containment
        assert!(box1.contains(50.0, 50.0));
        assert!(!box1.contains(150.0, 50.0));
    }
}
