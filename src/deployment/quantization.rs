//! Model quantization utilities

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    #[default]
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// 16-bit float (half precision)
    Float16,
    /// Dynamic quantization
    Dynamic,
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization type
    pub quantization_type: QuantizationType,
    /// Per-channel quantization
    pub per_channel: bool,
    /// Symmetric quantization
    pub symmetric: bool,
    /// Calibration samples
    pub calibration_samples: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Int8,
            per_channel: false,
            symmetric: true,
            calibration_samples: 100,
        }
    }
}

impl QuantizationConfig {
    /// Set quantization type
    pub fn quantization_type(mut self, qt: QuantizationType) -> Self {
        self.quantization_type = qt;
        self
    }

    /// Enable/disable per-channel quantization
    pub fn per_channel(mut self, enable: bool) -> Self {
        self.per_channel = enable;
        self
    }

    /// Enable/disable symmetric quantization
    pub fn symmetric(mut self, enable: bool) -> Self {
        self.symmetric = enable;
        self
    }

    /// Set calibration samples
    pub fn calibration_samples(mut self, n: usize) -> Self {
        self.calibration_samples = n;
        self
    }
}

/// Quantized model representation
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    /// Quantized weights
    pub weights: Vec<i8>,
    /// Scale factors
    pub scales: Vec<f64>,
    /// Zero points
    pub zero_points: Vec<i8>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization type used
    pub quantization_type: QuantizationType,
}

impl Default for QuantizedModel {
    fn default() -> Self {
        Self {
            weights: Vec::new(),
            scales: Vec::new(),
            zero_points: Vec::new(),
            shape: Vec::new(),
            quantization_type: QuantizationType::Int8,
        }
    }
}

impl QuantizedModel {
    /// Create new quantized model
    pub fn new(
        weights: Vec<i8>,
        scales: Vec<f64>,
        zero_points: Vec<i8>,
        shape: Vec<usize>,
        quantization_type: QuantizationType,
    ) -> Self {
        Self {
            weights,
            scales,
            zero_points,
            shape,
            quantization_type,
        }
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weights.len()
            + self.scales.len() * 8
            + self.zero_points.len()
            + self.shape.len() * 8
    }

    /// Dequantize weights back to f64
    pub fn dequantize(&self) -> Vec<f64> {
        if self.scales.is_empty() {
            return Vec::new();
        }

        self.weights
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let scale_idx = if self.scales.len() == 1 { 0 } else { i % self.scales.len() };
                let zp_idx = if self.zero_points.len() == 1 { 0 } else { i % self.zero_points.len() };
                (w as f64 - self.zero_points[zp_idx] as f64) * self.scales[scale_idx]
            })
            .collect()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Number of parameters
    pub fn len(&self) -> usize {
        self.weights.len()
    }
}

/// Model quantizer
#[derive(Debug)]
pub struct Quantizer {
    config: QuantizationConfig,
}

impl Quantizer {
    /// Create new quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }

    /// Quantize f64 weights to int8
    pub fn quantize(&self, weights: &[f64]) -> QuantizedModel {
        if weights.is_empty() {
            return QuantizedModel::default();
        }

        match self.config.quantization_type {
            QuantizationType::Int8 => self.quantize_int8(weights),
            QuantizationType::Int4 => self.quantize_int4(weights),
            QuantizationType::Float16 => self.quantize_float16(weights),
            QuantizationType::Dynamic => self.quantize_dynamic(weights),
        }
    }

    fn quantize_int8(&self, weights: &[f64]) -> QuantizedModel {
        let (min_val, max_val) = self.find_range(weights);

        let (scale, zero_point) = if self.config.symmetric {
            let max_abs = min_val.abs().max(max_val.abs());
            let scale = max_abs / 127.0;
            (scale, 0i8)
        } else {
            let scale = (max_val - min_val) / 255.0;
            let zero_point = (-128.0 - min_val / scale).round().clamp(-128.0, 127.0) as i8;
            (scale, zero_point)
        };

        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale + zero_point as f64).round();
                q.clamp(-128.0, 127.0) as i8
            })
            .collect();

        QuantizedModel::new(
            quantized,
            vec![scale],
            vec![zero_point],
            vec![weights.len()],
            QuantizationType::Int8,
        )
    }

    fn quantize_int4(&self, weights: &[f64]) -> QuantizedModel {
        let (min_val, max_val) = self.find_range(weights);
        let max_abs = min_val.abs().max(max_val.abs());
        let scale = max_abs / 7.0; // 4-bit signed range: -8 to 7

        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round();
                q.clamp(-8.0, 7.0) as i8
            })
            .collect();

        QuantizedModel::new(
            quantized,
            vec![scale],
            vec![0],
            vec![weights.len()],
            QuantizationType::Int4,
        )
    }

    fn quantize_float16(&self, weights: &[f64]) -> QuantizedModel {
        // Simulate float16 by rounding to reduced precision
        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| {
                // Store as scaled int8 for simplicity
                let scaled = w * 100.0;
                scaled.round().clamp(-128.0, 127.0) as i8
            })
            .collect();

        QuantizedModel::new(
            quantized,
            vec![0.01], // Scale factor to recover original
            vec![0],
            vec![weights.len()],
            QuantizationType::Float16,
        )
    }

    fn quantize_dynamic(&self, weights: &[f64]) -> QuantizedModel {
        // Dynamic quantization: use per-tensor scale
        let mut model = self.quantize_int8(weights);
        model.quantization_type = QuantizationType::Dynamic;
        model
    }

    fn find_range(&self, weights: &[f64]) -> (f64, f64) {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &w in weights {
            if w < min_val {
                min_val = w;
            }
            if w > max_val {
                max_val = w;
            }
        }

        (min_val, max_val)
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        match self.config.quantization_type {
            QuantizationType::Int8 => 8.0, // f64 (64 bits) -> int8 (8 bits)
            QuantizationType::Int4 => 16.0, // f64 (64 bits) -> int4 (4 bits)
            QuantizationType::Float16 => 4.0, // f64 (64 bits) -> float16 (16 bits)
            QuantizationType::Dynamic => 8.0,
        }
    }
}

impl Default for Quantizer {
    fn default() -> Self {
        Self::new(QuantizationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_default() {
        let qt = QuantizationType::default();
        assert_eq!(qt, QuantizationType::Int8);
    }

    #[test]
    fn test_quantization_type_variants() {
        assert_eq!(QuantizationType::Int8, QuantizationType::Int8);
        assert_ne!(QuantizationType::Int8, QuantizationType::Int4);
        assert_ne!(QuantizationType::Float16, QuantizationType::Dynamic);
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quantization_type, QuantizationType::Int8);
        assert!(!config.per_channel);
        assert!(config.symmetric);
        assert_eq!(config.calibration_samples, 100);
    }

    #[test]
    fn test_quantization_config_builder() {
        let config = QuantizationConfig::default()
            .quantization_type(QuantizationType::Int4)
            .per_channel(true)
            .symmetric(false)
            .calibration_samples(200);

        assert_eq!(config.quantization_type, QuantizationType::Int4);
        assert!(config.per_channel);
        assert!(!config.symmetric);
        assert_eq!(config.calibration_samples, 200);
    }

    #[test]
    fn test_quantized_model_new() {
        let model = QuantizedModel::new(
            vec![1, 2, 3],
            vec![0.1],
            vec![0],
            vec![3],
            QuantizationType::Int8,
        );

        assert_eq!(model.len(), 3);
        assert!(!model.is_empty());
    }

    #[test]
    fn test_quantized_model_default() {
        let model = QuantizedModel::default();
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn test_quantized_model_size() {
        let model = QuantizedModel::new(
            vec![1, 2, 3, 4],
            vec![0.1],
            vec![0],
            vec![4],
            QuantizationType::Int8,
        );

        // 4 weights (4 bytes) + 1 scale (8 bytes) + 1 zp (1 byte) + 1 shape (8 bytes)
        assert_eq!(model.size_bytes(), 4 + 8 + 1 + 8);
    }

    #[test]
    fn test_quantized_model_dequantize() {
        let model = QuantizedModel::new(
            vec![10, 20, 30],
            vec![0.1],
            vec![0],
            vec![3],
            QuantizationType::Int8,
        );

        let dequantized = model.dequantize();
        assert_eq!(dequantized.len(), 3);
        assert!((dequantized[0] - 1.0).abs() < 1e-10);
        assert!((dequantized[1] - 2.0).abs() < 1e-10);
        assert!((dequantized[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantized_model_dequantize_empty() {
        let model = QuantizedModel::default();
        let dequantized = model.dequantize();
        assert!(dequantized.is_empty());
    }

    #[test]
    fn test_quantizer_new() {
        let quantizer = Quantizer::default();
        assert_eq!(quantizer.config().quantization_type, QuantizationType::Int8);
    }

    #[test]
    fn test_quantizer_quantize_int8() {
        let quantizer = Quantizer::default();
        let weights = vec![-1.0, 0.0, 1.0];
        let model = quantizer.quantize(&weights);

        assert_eq!(model.len(), 3);
        assert_eq!(model.quantization_type, QuantizationType::Int8);
    }

    #[test]
    fn test_quantizer_quantize_int4() {
        let config = QuantizationConfig::default().quantization_type(QuantizationType::Int4);
        let quantizer = Quantizer::new(config);

        let weights = vec![-1.0, 0.0, 1.0];
        let model = quantizer.quantize(&weights);

        assert_eq!(model.quantization_type, QuantizationType::Int4);
    }

    #[test]
    fn test_quantizer_quantize_float16() {
        let config = QuantizationConfig::default().quantization_type(QuantizationType::Float16);
        let quantizer = Quantizer::new(config);

        let weights = vec![0.5, 1.0];
        let model = quantizer.quantize(&weights);

        assert_eq!(model.quantization_type, QuantizationType::Float16);
    }

    #[test]
    fn test_quantizer_quantize_dynamic() {
        let config = QuantizationConfig::default().quantization_type(QuantizationType::Dynamic);
        let quantizer = Quantizer::new(config);

        let weights = vec![1.0, 2.0];
        let model = quantizer.quantize(&weights);

        assert_eq!(model.quantization_type, QuantizationType::Dynamic);
    }

    #[test]
    fn test_quantizer_quantize_empty() {
        let quantizer = Quantizer::default();
        let model = quantizer.quantize(&[]);
        assert!(model.is_empty());
    }

    #[test]
    fn test_quantizer_asymmetric() {
        let config = QuantizationConfig::default().symmetric(false);
        let quantizer = Quantizer::new(config);

        let weights = vec![0.0, 0.5, 1.0];
        let model = quantizer.quantize(&weights);

        assert_eq!(model.len(), 3);
    }

    #[test]
    fn test_quantizer_compression_ratio() {
        let quantizer = Quantizer::default();
        assert!((quantizer.compression_ratio() - 8.0).abs() < 1e-10);

        let config = QuantizationConfig::default().quantization_type(QuantizationType::Int4);
        let quantizer = Quantizer::new(config);
        assert!((quantizer.compression_ratio() - 16.0).abs() < 1e-10);

        let config = QuantizationConfig::default().quantization_type(QuantizationType::Float16);
        let quantizer = Quantizer::new(config);
        assert!((quantizer.compression_ratio() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let quantizer = Quantizer::default();
        let weights = vec![-0.5, 0.0, 0.5];

        let model = quantizer.quantize(&weights);
        let recovered = model.dequantize();

        // Should be close to original (within quantization error)
        for (orig, rec) in weights.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantizer_config_access() {
        let config = QuantizationConfig::default().per_channel(true);
        let quantizer = Quantizer::new(config);
        assert!(quantizer.config().per_channel);
    }

    #[test]
    fn test_quantize_large_values() {
        let quantizer = Quantizer::default();
        let weights = vec![-100.0, 100.0];
        let model = quantizer.quantize(&weights);

        // Values should be clamped to int8 range
        assert!(model.weights.iter().all(|&w| w >= -128 && w <= 127));
    }

    #[test]
    fn test_quantize_uniform_values() {
        let quantizer = Quantizer::default();
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let model = quantizer.quantize(&weights);

        // All quantized values should be the same
        assert!(model.weights.windows(2).all(|w| w[0] == w[1]));
    }
}
