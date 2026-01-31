//! Vision Processing
//!
//! Image processing and visual feature extraction.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.multimodal import ImageFormat, VisionConfig, process_image
//! config = VisionConfig(size=224, normalize=True)
//! ```

/// Image format type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFormat {
    /// RGB format (default)
    #[default]
    Rgb,
    /// BGR format
    Bgr,
    /// Grayscale
    Grayscale,
    /// RGBA with alpha
    Rgba,
}

impl ImageFormat {
    /// Get format name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Bgr => "bgr",
            Self::Grayscale => "grayscale",
            Self::Rgba => "rgba",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rgb" => Some(Self::Rgb),
            "bgr" => Some(Self::Bgr),
            "grayscale" | "gray" | "grey" => Some(Self::Grayscale),
            "rgba" => Some(Self::Rgba),
            _ => None,
        }
    }

    /// Get number of channels
    pub fn channels(&self) -> usize {
        match self {
            Self::Rgb | Self::Bgr => 3,
            Self::Grayscale => 1,
            Self::Rgba => 4,
        }
    }
}

/// Resize interpolation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Bilinear (default)
    #[default]
    Bilinear,
    /// Nearest neighbor
    Nearest,
    /// Bicubic
    Bicubic,
    /// Lanczos
    Lanczos,
}

impl Interpolation {
    /// Get method name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bilinear => "bilinear",
            Self::Nearest => "nearest",
            Self::Bicubic => "bicubic",
            Self::Lanczos => "lanczos",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bilinear" => Some(Self::Bilinear),
            "nearest" | "nn" => Some(Self::Nearest),
            "bicubic" => Some(Self::Bicubic),
            "lanczos" => Some(Self::Lanczos),
            _ => None,
        }
    }
}

/// Vision model architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisionModel {
    /// Vision Transformer (default)
    #[default]
    ViT,
    /// ResNet
    ResNet,
    /// CLIP vision encoder
    Clip,
    /// DINOv2
    DINOv2,
    /// ConvNeXt
    ConvNeXt,
    /// Swin Transformer
    Swin,
}

impl VisionModel {
    /// Get model name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ViT => "vit",
            Self::ResNet => "resnet",
            Self::Clip => "clip",
            Self::DINOv2 => "dinov2",
            Self::ConvNeXt => "convnext",
            Self::Swin => "swin",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "vit" | "vision_transformer" => Some(Self::ViT),
            "resnet" => Some(Self::ResNet),
            "clip" => Some(Self::Clip),
            "dinov2" | "dino" => Some(Self::DINOv2),
            "convnext" => Some(Self::ConvNeXt),
            "swin" | "swin_transformer" => Some(Self::Swin),
            _ => None,
        }
    }

    /// List all models
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::ViT,
            Self::ResNet,
            Self::Clip,
            Self::DINOv2,
            Self::ConvNeXt,
            Self::Swin,
        ]
    }

    /// Get default input size
    pub fn default_size(&self) -> usize {
        match self {
            Self::ViT | Self::DINOv2 | Self::Swin => 224,
            Self::ResNet | Self::ConvNeXt => 224,
            Self::Clip => 224,
        }
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::ViT => 768,
            Self::ResNet => 2048,
            Self::Clip => 512,
            Self::DINOv2 => 768,
            Self::ConvNeXt => 1024,
            Self::Swin => 768,
        }
    }
}

/// Vision configuration
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Input image size (square)
    pub size: usize,
    /// Image format
    pub format: ImageFormat,
    /// Resize interpolation
    pub interpolation: Interpolation,
    /// Normalize values
    pub normalize: bool,
    /// Mean for normalization (per channel)
    pub mean: Vec<f32>,
    /// Std for normalization (per channel)
    pub std: Vec<f32>,
    /// Model architecture
    pub model: VisionModel,
    /// Enable center crop
    pub center_crop: bool,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            size: 224,
            format: ImageFormat::Rgb,
            interpolation: Interpolation::Bilinear,
            normalize: true,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            model: VisionModel::ViT,
            center_crop: true,
        }
    }
}

impl VisionConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// ImageNet preset
    pub fn imagenet() -> Self {
        Self::default()
    }

    /// CLIP preset
    pub fn clip() -> Self {
        Self {
            size: 224,
            format: ImageFormat::Rgb,
            interpolation: Interpolation::Bicubic,
            normalize: true,
            mean: vec![0.48145466, 0.4578275, 0.40821073],
            std: vec![0.26862954, 0.261_302_6, 0.275_777_1],
            model: VisionModel::Clip,
            center_crop: true,
        }
    }

    /// Set size
    pub fn size(mut self, s: usize) -> Self {
        self.size = s;
        self
    }

    /// Set format
    pub fn format(mut self, f: ImageFormat) -> Self {
        self.format = f;
        self
    }

    /// Set interpolation
    pub fn interpolation(mut self, i: Interpolation) -> Self {
        self.interpolation = i;
        self
    }

    /// Set normalize
    pub fn normalize(mut self, enabled: bool) -> Self {
        self.normalize = enabled;
        self
    }

    /// Set mean
    pub fn mean(mut self, m: Vec<f32>) -> Self {
        self.mean = m;
        self
    }

    /// Set std
    pub fn std(mut self, s: Vec<f32>) -> Self {
        self.std = s;
        self
    }

    /// Set model
    pub fn model(mut self, m: VisionModel) -> Self {
        self.model = m;
        self
    }

    /// Set center crop
    pub fn center_crop(mut self, enabled: bool) -> Self {
        self.center_crop = enabled;
        self
    }

    /// Get input shape (C, H, W)
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (self.format.channels(), self.size, self.size)
    }

    /// Calculate input tensor size
    pub fn tensor_size(&self) -> usize {
        let (c, h, w) = self.input_shape();
        c * h * w
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.size == 0 {
            return Err("Image size must be > 0".to_string());
        }
        if self.normalize && self.mean.len() != self.format.channels() {
            return Err("Mean length must match number of channels".to_string());
        }
        if self.normalize && self.std.len() != self.format.channels() {
            return Err("Std length must match number of channels".to_string());
        }
        Ok(())
    }
}

/// Image features result
#[derive(Debug, Clone)]
pub struct ImageFeatures {
    /// Feature dimension
    pub feature_dim: usize,
    /// Number of patches (for ViT-style models)
    pub num_patches: usize,
    /// Original image width
    pub original_width: usize,
    /// Original image height
    pub original_height: usize,
    /// Processing time in ms
    pub processing_time_ms: u64,
}

impl ImageFeatures {
    /// Create new features
    pub fn new(feature_dim: usize) -> Self {
        Self {
            feature_dim,
            num_patches: 0,
            original_width: 0,
            original_height: 0,
            processing_time_ms: 0,
        }
    }

    /// Set num patches
    pub fn num_patches(mut self, n: usize) -> Self {
        self.num_patches = n;
        self
    }

    /// Set original size
    pub fn original_size(mut self, width: usize, height: usize) -> Self {
        self.original_width = width;
        self.original_height = height;
        self
    }

    /// Set processing time
    pub fn processing_time(mut self, ms: u64) -> Self {
        self.processing_time_ms = ms;
        self
    }

    /// Get aspect ratio of original image
    pub fn aspect_ratio(&self) -> f32 {
        if self.original_height == 0 {
            return 1.0;
        }
        self.original_width as f32 / self.original_height as f32
    }

    /// Estimate memory for features (f32)
    pub fn memory_bytes(&self) -> usize {
        self.feature_dim * 4
    }
}

/// Vision processing statistics
#[derive(Debug, Clone, Default)]
pub struct VisionStats {
    /// Images processed
    pub images_processed: usize,
    /// Total processing time (ms)
    pub total_time_ms: u64,
    /// Total pixels processed
    pub total_pixels: usize,
}

impl VisionStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record image processing
    pub fn record(&mut self, features: &ImageFeatures, config: &VisionConfig) {
        self.images_processed += 1;
        self.total_time_ms += features.processing_time_ms;
        self.total_pixels += config.size * config.size;
    }

    /// Get average processing time
    pub fn avg_time_ms(&self) -> f64 {
        if self.images_processed == 0 {
            return 0.0;
        }
        self.total_time_ms as f64 / self.images_processed as f64
    }

    /// Get throughput (images per second)
    pub fn throughput(&self) -> f64 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        self.images_processed as f64 / (self.total_time_ms as f64 / 1000.0)
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Vision: {} images, {:.1}ms avg, {:.1} img/s",
            self.images_processed,
            self.avg_time_ms(),
            self.throughput()
        )
    }
}

/// Calculate number of patches for ViT
pub fn calculate_num_patches(image_size: usize, patch_size: usize) -> usize {
    if patch_size == 0 {
        return 0;
    }
    (image_size / patch_size) * (image_size / patch_size)
}

/// Estimate processing time in ms
pub fn estimate_processing_time(config: &VisionConfig) -> u64 {
    let base_time = (config.size * config.size / 1000) as u64;
    let model_factor = match config.model {
        VisionModel::ViT => 2,
        VisionModel::ResNet => 1,
        VisionModel::Clip => 2,
        VisionModel::DINOv2 => 3,
        VisionModel::ConvNeXt => 2,
        VisionModel::Swin => 3,
    };
    base_time * model_factor + 5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_default() {
        assert_eq!(ImageFormat::default(), ImageFormat::Rgb);
    }

    #[test]
    fn test_image_format_as_str() {
        assert_eq!(ImageFormat::Rgb.as_str(), "rgb");
        assert_eq!(ImageFormat::Bgr.as_str(), "bgr");
        assert_eq!(ImageFormat::Grayscale.as_str(), "grayscale");
        assert_eq!(ImageFormat::Rgba.as_str(), "rgba");
    }

    #[test]
    fn test_image_format_parse() {
        assert_eq!(ImageFormat::parse("rgb"), Some(ImageFormat::Rgb));
        assert_eq!(ImageFormat::parse("bgr"), Some(ImageFormat::Bgr));
        assert_eq!(ImageFormat::parse("gray"), Some(ImageFormat::Grayscale));
        assert_eq!(ImageFormat::parse("grey"), Some(ImageFormat::Grayscale));
        assert_eq!(ImageFormat::parse("rgba"), Some(ImageFormat::Rgba));
        assert_eq!(ImageFormat::parse("unknown"), None);
    }

    #[test]
    fn test_image_format_channels() {
        assert_eq!(ImageFormat::Rgb.channels(), 3);
        assert_eq!(ImageFormat::Bgr.channels(), 3);
        assert_eq!(ImageFormat::Grayscale.channels(), 1);
        assert_eq!(ImageFormat::Rgba.channels(), 4);
    }

    #[test]
    fn test_interpolation_as_str() {
        assert_eq!(Interpolation::Bilinear.as_str(), "bilinear");
        assert_eq!(Interpolation::Nearest.as_str(), "nearest");
        assert_eq!(Interpolation::Bicubic.as_str(), "bicubic");
        assert_eq!(Interpolation::Lanczos.as_str(), "lanczos");
    }

    #[test]
    fn test_interpolation_parse() {
        assert_eq!(Interpolation::parse("bilinear"), Some(Interpolation::Bilinear));
        assert_eq!(Interpolation::parse("nn"), Some(Interpolation::Nearest));
        assert_eq!(Interpolation::parse("bicubic"), Some(Interpolation::Bicubic));
        assert_eq!(Interpolation::parse("lanczos"), Some(Interpolation::Lanczos));
        assert_eq!(Interpolation::parse("unknown"), None);
    }

    #[test]
    fn test_vision_model_as_str() {
        assert_eq!(VisionModel::ViT.as_str(), "vit");
        assert_eq!(VisionModel::ResNet.as_str(), "resnet");
        assert_eq!(VisionModel::Clip.as_str(), "clip");
        assert_eq!(VisionModel::DINOv2.as_str(), "dinov2");
        assert_eq!(VisionModel::ConvNeXt.as_str(), "convnext");
        assert_eq!(VisionModel::Swin.as_str(), "swin");
    }

    #[test]
    fn test_vision_model_parse() {
        assert_eq!(VisionModel::parse("vit"), Some(VisionModel::ViT));
        assert_eq!(VisionModel::parse("vision_transformer"), Some(VisionModel::ViT));
        assert_eq!(VisionModel::parse("resnet"), Some(VisionModel::ResNet));
        assert_eq!(VisionModel::parse("clip"), Some(VisionModel::Clip));
        assert_eq!(VisionModel::parse("dino"), Some(VisionModel::DINOv2));
        assert_eq!(VisionModel::parse("convnext"), Some(VisionModel::ConvNeXt));
        assert_eq!(VisionModel::parse("swin_transformer"), Some(VisionModel::Swin));
        assert_eq!(VisionModel::parse("unknown"), None);
    }

    #[test]
    fn test_vision_model_list_all() {
        assert_eq!(VisionModel::list_all().len(), 6);
    }

    #[test]
    fn test_vision_model_default_size() {
        assert_eq!(VisionModel::ViT.default_size(), 224);
        assert_eq!(VisionModel::Clip.default_size(), 224);
    }

    #[test]
    fn test_vision_model_embed_dim() {
        assert_eq!(VisionModel::ViT.embed_dim(), 768);
        assert_eq!(VisionModel::ResNet.embed_dim(), 2048);
        assert_eq!(VisionModel::Clip.embed_dim(), 512);
    }

    #[test]
    fn test_vision_config_default() {
        let config = VisionConfig::default();
        assert_eq!(config.size, 224);
        assert_eq!(config.format, ImageFormat::Rgb);
        assert!(config.normalize);
    }

    #[test]
    fn test_vision_config_imagenet() {
        let config = VisionConfig::imagenet();
        assert_eq!(config.size, 224);
        assert_eq!(config.mean.len(), 3);
        assert_eq!(config.std.len(), 3);
    }

    #[test]
    fn test_vision_config_clip() {
        let config = VisionConfig::clip();
        assert_eq!(config.model, VisionModel::Clip);
        assert_eq!(config.interpolation, Interpolation::Bicubic);
    }

    #[test]
    fn test_vision_config_builder() {
        let config = VisionConfig::new()
            .size(384)
            .format(ImageFormat::Bgr)
            .interpolation(Interpolation::Bicubic)
            .normalize(false)
            .model(VisionModel::DINOv2)
            .center_crop(false);

        assert_eq!(config.size, 384);
        assert_eq!(config.format, ImageFormat::Bgr);
        assert_eq!(config.interpolation, Interpolation::Bicubic);
        assert!(!config.normalize);
        assert_eq!(config.model, VisionModel::DINOv2);
        assert!(!config.center_crop);
    }

    #[test]
    fn test_vision_config_input_shape() {
        let config = VisionConfig::default();
        assert_eq!(config.input_shape(), (3, 224, 224));

        let gray_config = VisionConfig::new().format(ImageFormat::Grayscale);
        assert_eq!(gray_config.input_shape(), (1, 224, 224));
    }

    #[test]
    fn test_vision_config_tensor_size() {
        let config = VisionConfig::default();
        assert_eq!(config.tensor_size(), 3 * 224 * 224);
    }

    #[test]
    fn test_vision_config_validate() {
        let valid = VisionConfig::default();
        assert!(valid.validate().is_ok());

        let zero_size = VisionConfig::new().size(0);
        assert!(zero_size.validate().is_err());

        let bad_mean = VisionConfig::new().mean(vec![0.5]); // Only 1 value for RGB
        assert!(bad_mean.validate().is_err());

        let bad_std = VisionConfig::new().std(vec![0.5, 0.5]); // Only 2 values
        assert!(bad_std.validate().is_err());
    }

    #[test]
    fn test_image_features_new() {
        let features = ImageFeatures::new(768);
        assert_eq!(features.feature_dim, 768);
    }

    #[test]
    fn test_image_features_builder() {
        let features = ImageFeatures::new(512)
            .num_patches(196)
            .original_size(640, 480)
            .processing_time(25);

        assert_eq!(features.num_patches, 196);
        assert_eq!(features.original_width, 640);
        assert_eq!(features.original_height, 480);
        assert_eq!(features.processing_time_ms, 25);
    }

    #[test]
    fn test_image_features_aspect_ratio() {
        let features = ImageFeatures::new(768).original_size(1920, 1080);
        let ratio = features.aspect_ratio();
        assert!((ratio - 1.778).abs() < 0.01); // 16:9
    }

    #[test]
    fn test_image_features_memory() {
        let features = ImageFeatures::new(768);
        assert_eq!(features.memory_bytes(), 768 * 4);
    }

    #[test]
    fn test_vision_stats_new() {
        let stats = VisionStats::new();
        assert_eq!(stats.images_processed, 0);
    }

    #[test]
    fn test_vision_stats_record() {
        let mut stats = VisionStats::new();
        let config = VisionConfig::default();
        let features = ImageFeatures::new(768).processing_time(20);

        stats.record(&features, &config);
        stats.record(&features, &config);

        assert_eq!(stats.images_processed, 2);
        assert_eq!(stats.total_time_ms, 40);
        assert_eq!(stats.total_pixels, 224 * 224 * 2);
    }

    #[test]
    fn test_vision_stats_avg_time() {
        let mut stats = VisionStats::new();
        let config = VisionConfig::default();
        let features = ImageFeatures::new(768).processing_time(30);

        stats.record(&features, &config);
        stats.record(&features, &config);

        assert!((stats.avg_time_ms() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_vision_stats_throughput() {
        let mut stats = VisionStats::new();
        let config = VisionConfig::default();
        let features = ImageFeatures::new(768).processing_time(100); // 100ms per image

        stats.record(&features, &config);
        stats.record(&features, &config);

        assert!((stats.throughput() - 10.0).abs() < 0.1); // 10 images per second
    }

    #[test]
    fn test_vision_stats_format() {
        let mut stats = VisionStats::new();
        let config = VisionConfig::default();
        let features = ImageFeatures::new(768).processing_time(25);

        stats.record(&features, &config);

        let formatted = stats.format();
        assert!(formatted.contains("1 images"));
        assert!(formatted.contains("25.0ms"));
    }

    #[test]
    fn test_calculate_num_patches() {
        assert_eq!(calculate_num_patches(224, 16), 196); // (224/16)^2 = 14^2 = 196
        assert_eq!(calculate_num_patches(384, 16), 576); // (384/16)^2 = 24^2 = 576
        assert_eq!(calculate_num_patches(224, 0), 0);
    }

    #[test]
    fn test_estimate_processing_time() {
        let vit_config = VisionConfig::new().model(VisionModel::ViT);
        let resnet_config = VisionConfig::new().model(VisionModel::ResNet);

        let vit_time = estimate_processing_time(&vit_config);
        let resnet_time = estimate_processing_time(&resnet_config);

        assert!(vit_time > 0);
        assert!(resnet_time > 0);
        assert!(vit_time > resnet_time); // ViT typically slower
    }
}
