//! Audio Feature Extraction
//!
//! Mel spectrograms, MFCCs, and audio preprocessing.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.audio import FeatureType, AudioConfig, extract_features
//! features = extract_features(audio, config=AudioConfig(sample_rate=16000))
//! ```

/// Audio feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureType {
    /// Mel spectrogram (default)
    #[default]
    MelSpectrogram,
    /// Mel-frequency cepstral coefficients
    Mfcc,
    /// Raw waveform
    Waveform,
    /// Log mel spectrogram
    LogMel,
    /// Filter bank features
    FilterBank,
    /// Spectrogram
    Spectrogram,
}

impl FeatureType {
    /// Get feature name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MelSpectrogram => "mel_spectrogram",
            Self::Mfcc => "mfcc",
            Self::Waveform => "waveform",
            Self::LogMel => "log_mel",
            Self::FilterBank => "filter_bank",
            Self::Spectrogram => "spectrogram",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "mel_spectrogram" | "mel" => Some(Self::MelSpectrogram),
            "mfcc" => Some(Self::Mfcc),
            "waveform" | "raw" => Some(Self::Waveform),
            "log_mel" | "logmel" => Some(Self::LogMel),
            "filter_bank" | "fbank" => Some(Self::FilterBank),
            "spectrogram" | "spec" => Some(Self::Spectrogram),
            _ => None,
        }
    }

    /// List all types
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::MelSpectrogram,
            Self::Mfcc,
            Self::Waveform,
            Self::LogMel,
            Self::FilterBank,
            Self::Spectrogram,
        ]
    }

    /// Get output dimensions for feature type
    pub fn output_dims(&self, n_mels: usize, n_mfcc: usize) -> usize {
        match self {
            Self::MelSpectrogram | Self::LogMel | Self::FilterBank => n_mels,
            Self::Mfcc => n_mfcc,
            Self::Waveform => 1,
            Self::Spectrogram => 257, // Default FFT size / 2 + 1
        }
    }
}

/// Window function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowType {
    /// Hann window (default)
    #[default]
    Hann,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Rectangular window
    Rectangular,
    /// Bartlett window
    Bartlett,
}

impl WindowType {
    /// Get window name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hann => "hann",
            Self::Hamming => "hamming",
            Self::Blackman => "blackman",
            Self::Rectangular => "rectangular",
            Self::Bartlett => "bartlett",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hann" | "hanning" => Some(Self::Hann),
            "hamming" => Some(Self::Hamming),
            "blackman" => Some(Self::Blackman),
            "rectangular" | "rect" | "boxcar" => Some(Self::Rectangular),
            "bartlett" | "triangular" => Some(Self::Bartlett),
            _ => None,
        }
    }
}

/// Audio configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of mel filterbanks
    pub n_mels: usize,
    /// Number of MFCC coefficients
    pub n_mfcc: usize,
    /// FFT window size
    pub n_fft: usize,
    /// Hop length (samples between frames)
    pub hop_length: usize,
    /// Window type
    pub window: WindowType,
    /// Feature type
    pub feature_type: FeatureType,
    /// Normalize features
    pub normalize: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_mfcc: 13,
            n_fft: 400,
            hop_length: 160,
            window: WindowType::Hann,
            feature_type: FeatureType::MelSpectrogram,
            normalize: true,
        }
    }
}

impl AudioConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Whisper-compatible config (16kHz, 80 mels)
    pub fn whisper() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_mfcc: 13,
            n_fft: 400,
            hop_length: 160,
            window: WindowType::Hann,
            feature_type: FeatureType::LogMel,
            normalize: true,
        }
    }

    /// Wav2Vec-compatible config
    pub fn wav2vec() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_mfcc: 13,
            n_fft: 400,
            hop_length: 320,
            window: WindowType::Hann,
            feature_type: FeatureType::Waveform,
            normalize: true,
        }
    }

    /// Set sample rate
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set number of mel bands
    pub fn n_mels(mut self, n: usize) -> Self {
        self.n_mels = n;
        self
    }

    /// Set number of MFCC coefficients
    pub fn n_mfcc(mut self, n: usize) -> Self {
        self.n_mfcc = n;
        self
    }

    /// Set FFT size
    pub fn n_fft(mut self, n: usize) -> Self {
        self.n_fft = n;
        self
    }

    /// Set hop length
    pub fn hop_length(mut self, n: usize) -> Self {
        self.hop_length = n;
        self
    }

    /// Set window type
    pub fn window(mut self, w: WindowType) -> Self {
        self.window = w;
        self
    }

    /// Set feature type
    pub fn feature_type(mut self, f: FeatureType) -> Self {
        self.feature_type = f;
        self
    }

    /// Set normalization
    pub fn normalize(mut self, enabled: bool) -> Self {
        self.normalize = enabled;
        self
    }

    /// Calculate frame duration in ms
    pub fn frame_duration_ms(&self) -> f32 {
        self.n_fft as f32 / self.sample_rate as f32 * 1000.0
    }

    /// Calculate hop duration in ms
    pub fn hop_duration_ms(&self) -> f32 {
        self.hop_length as f32 / self.sample_rate as f32 * 1000.0
    }

    /// Calculate number of frames for audio duration
    pub fn num_frames(&self, duration_secs: f32) -> usize {
        let samples = (duration_secs * self.sample_rate as f32) as usize;
        if samples <= self.n_fft {
            1
        } else {
            (samples - self.n_fft) / self.hop_length + 1
        }
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0 {
            return Err("Sample rate must be > 0".to_string());
        }
        if self.n_fft == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if self.hop_length == 0 {
            return Err("Hop length must be > 0".to_string());
        }
        if self.hop_length > self.n_fft {
            return Err("Hop length must be <= FFT size".to_string());
        }
        if self.n_mels == 0 {
            return Err("Number of mel bands must be > 0".to_string());
        }
        Ok(())
    }
}

/// Audio features result
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// Feature type
    pub feature_type: FeatureType,
    /// Number of frames
    pub num_frames: usize,
    /// Feature dimension
    pub feature_dim: usize,
    /// Duration in seconds
    pub duration_secs: f32,
    /// Sample rate
    pub sample_rate: u32,
}

impl AudioFeatures {
    /// Create new features
    pub fn new(feature_type: FeatureType, num_frames: usize, feature_dim: usize) -> Self {
        Self {
            feature_type,
            num_frames,
            feature_dim,
            duration_secs: 0.0,
            sample_rate: 16000,
        }
    }

    /// Set duration
    pub fn duration(mut self, secs: f32) -> Self {
        self.duration_secs = secs;
        self
    }

    /// Set sample rate
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Get total feature count
    pub fn total_features(&self) -> usize {
        self.num_frames * self.feature_dim
    }

    /// Estimate memory size in bytes (f32)
    pub fn memory_bytes(&self) -> usize {
        self.total_features() * 4
    }
}

/// Audio preprocessing statistics
#[derive(Debug, Clone, Default)]
pub struct AudioStats {
    /// Total audio processed (seconds)
    pub total_duration_secs: f32,
    /// Total frames extracted
    pub total_frames: usize,
    /// Files processed
    pub files_processed: usize,
    /// Average duration per file
    pub avg_duration_secs: f32,
}

impl AudioStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record audio processing
    pub fn record(&mut self, features: &AudioFeatures) {
        self.total_duration_secs += features.duration_secs;
        self.total_frames += features.num_frames;
        self.files_processed += 1;
        self.avg_duration_secs = self.total_duration_secs / self.files_processed as f32;
    }

    /// Get frames per second
    pub fn frames_per_second(&self) -> f32 {
        if self.total_duration_secs == 0.0 {
            return 0.0;
        }
        self.total_frames as f32 / self.total_duration_secs
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Audio: {:.1}s total, {} frames, {} files ({:.1}s avg)",
            self.total_duration_secs,
            self.total_frames,
            self.files_processed,
            self.avg_duration_secs
        )
    }
}

/// Calculate mel frequency from Hz
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Calculate Hz from mel frequency
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Calculate number of frames for given samples
pub fn calculate_num_frames(num_samples: usize, n_fft: usize, hop_length: usize) -> usize {
    if num_samples <= n_fft {
        1
    } else {
        (num_samples - n_fft) / hop_length + 1
    }
}

/// Estimate feature extraction time in ms
pub fn estimate_extraction_time(duration_secs: f32, feature_type: FeatureType) -> u64 {
    let base_time = (duration_secs * 10.0) as u64; // 10ms per second base
    match feature_type {
        FeatureType::Waveform => base_time,
        FeatureType::Spectrogram => base_time * 2,
        FeatureType::MelSpectrogram | FeatureType::LogMel => base_time * 3,
        FeatureType::FilterBank => base_time * 3,
        FeatureType::Mfcc => base_time * 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_type_default() {
        assert_eq!(FeatureType::default(), FeatureType::MelSpectrogram);
    }

    #[test]
    fn test_feature_type_as_str() {
        assert_eq!(FeatureType::MelSpectrogram.as_str(), "mel_spectrogram");
        assert_eq!(FeatureType::Mfcc.as_str(), "mfcc");
        assert_eq!(FeatureType::Waveform.as_str(), "waveform");
        assert_eq!(FeatureType::LogMel.as_str(), "log_mel");
        assert_eq!(FeatureType::FilterBank.as_str(), "filter_bank");
        assert_eq!(FeatureType::Spectrogram.as_str(), "spectrogram");
    }

    #[test]
    fn test_feature_type_parse() {
        assert_eq!(FeatureType::parse("mel"), Some(FeatureType::MelSpectrogram));
        assert_eq!(FeatureType::parse("mfcc"), Some(FeatureType::Mfcc));
        assert_eq!(FeatureType::parse("raw"), Some(FeatureType::Waveform));
        assert_eq!(FeatureType::parse("logmel"), Some(FeatureType::LogMel));
        assert_eq!(FeatureType::parse("fbank"), Some(FeatureType::FilterBank));
        assert_eq!(FeatureType::parse("spec"), Some(FeatureType::Spectrogram));
        assert_eq!(FeatureType::parse("unknown"), None);
    }

    #[test]
    fn test_feature_type_list_all() {
        assert_eq!(FeatureType::list_all().len(), 6);
    }

    #[test]
    fn test_feature_type_output_dims() {
        assert_eq!(FeatureType::MelSpectrogram.output_dims(80, 13), 80);
        assert_eq!(FeatureType::Mfcc.output_dims(80, 13), 13);
        assert_eq!(FeatureType::Waveform.output_dims(80, 13), 1);
        assert_eq!(FeatureType::Spectrogram.output_dims(80, 13), 257);
    }

    #[test]
    fn test_window_type_as_str() {
        assert_eq!(WindowType::Hann.as_str(), "hann");
        assert_eq!(WindowType::Hamming.as_str(), "hamming");
        assert_eq!(WindowType::Blackman.as_str(), "blackman");
        assert_eq!(WindowType::Rectangular.as_str(), "rectangular");
        assert_eq!(WindowType::Bartlett.as_str(), "bartlett");
    }

    #[test]
    fn test_window_type_parse() {
        assert_eq!(WindowType::parse("hanning"), Some(WindowType::Hann));
        assert_eq!(WindowType::parse("hamming"), Some(WindowType::Hamming));
        assert_eq!(WindowType::parse("blackman"), Some(WindowType::Blackman));
        assert_eq!(WindowType::parse("boxcar"), Some(WindowType::Rectangular));
        assert_eq!(WindowType::parse("triangular"), Some(WindowType::Bartlett));
        assert_eq!(WindowType::parse("unknown"), None);
    }

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
    }

    #[test]
    fn test_audio_config_whisper() {
        let config = AudioConfig::whisper();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.feature_type, FeatureType::LogMel);
    }

    #[test]
    fn test_audio_config_wav2vec() {
        let config = AudioConfig::wav2vec();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.feature_type, FeatureType::Waveform);
    }

    #[test]
    fn test_audio_config_builder() {
        let config = AudioConfig::new()
            .sample_rate(22050)
            .n_mels(128)
            .n_mfcc(20)
            .n_fft(512)
            .hop_length(256)
            .window(WindowType::Hamming)
            .feature_type(FeatureType::Mfcc)
            .normalize(false);

        assert_eq!(config.sample_rate, 22050);
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.n_mfcc, 20);
        assert_eq!(config.n_fft, 512);
        assert_eq!(config.hop_length, 256);
        assert_eq!(config.window, WindowType::Hamming);
        assert_eq!(config.feature_type, FeatureType::Mfcc);
        assert!(!config.normalize);
    }

    #[test]
    fn test_audio_config_frame_duration() {
        let config = AudioConfig::default();
        let frame_ms = config.frame_duration_ms();
        assert!((frame_ms - 25.0).abs() < 0.1); // 400/16000 = 25ms
    }

    #[test]
    fn test_audio_config_hop_duration() {
        let config = AudioConfig::default();
        let hop_ms = config.hop_duration_ms();
        assert!((hop_ms - 10.0).abs() < 0.1); // 160/16000 = 10ms
    }

    #[test]
    fn test_audio_config_num_frames() {
        let config = AudioConfig::default();
        let frames = config.num_frames(1.0); // 1 second
        assert!(frames > 0);
        // At 16kHz, 1s = 16000 samples, with hop 160: ~100 frames
        assert!(frames > 90 && frames < 110);
    }

    #[test]
    fn test_audio_config_validate() {
        let valid = AudioConfig::default();
        assert!(valid.validate().is_ok());

        let zero_rate = AudioConfig::new().sample_rate(0);
        assert!(zero_rate.validate().is_err());

        let zero_fft = AudioConfig::new().n_fft(0);
        assert!(zero_fft.validate().is_err());

        let zero_hop = AudioConfig::new().hop_length(0);
        assert!(zero_hop.validate().is_err());

        let bad_hop = AudioConfig::new().n_fft(256).hop_length(512);
        assert!(bad_hop.validate().is_err());

        let zero_mels = AudioConfig::new().n_mels(0);
        assert!(zero_mels.validate().is_err());
    }

    #[test]
    fn test_audio_features_new() {
        let features = AudioFeatures::new(FeatureType::MelSpectrogram, 100, 80);
        assert_eq!(features.num_frames, 100);
        assert_eq!(features.feature_dim, 80);
    }

    #[test]
    fn test_audio_features_builder() {
        let features = AudioFeatures::new(FeatureType::Mfcc, 50, 13)
            .duration(5.0)
            .sample_rate(22050);

        assert_eq!(features.duration_secs, 5.0);
        assert_eq!(features.sample_rate, 22050);
    }

    #[test]
    fn test_audio_features_total() {
        let features = AudioFeatures::new(FeatureType::MelSpectrogram, 100, 80);
        assert_eq!(features.total_features(), 8000);
    }

    #[test]
    fn test_audio_features_memory() {
        let features = AudioFeatures::new(FeatureType::MelSpectrogram, 100, 80);
        assert_eq!(features.memory_bytes(), 32000); // 8000 * 4 bytes
    }

    #[test]
    fn test_audio_stats_new() {
        let stats = AudioStats::new();
        assert_eq!(stats.files_processed, 0);
        assert_eq!(stats.total_duration_secs, 0.0);
    }

    #[test]
    fn test_audio_stats_record() {
        let mut stats = AudioStats::new();
        let features1 = AudioFeatures::new(FeatureType::MelSpectrogram, 100, 80).duration(10.0);
        let features2 = AudioFeatures::new(FeatureType::MelSpectrogram, 50, 80).duration(5.0);

        stats.record(&features1);
        stats.record(&features2);

        assert_eq!(stats.files_processed, 2);
        assert!((stats.total_duration_secs - 15.0).abs() < 0.01);
        assert!((stats.avg_duration_secs - 7.5).abs() < 0.01);
        assert_eq!(stats.total_frames, 150);
    }

    #[test]
    fn test_audio_stats_frames_per_second() {
        let mut stats = AudioStats::new();
        let features = AudioFeatures::new(FeatureType::MelSpectrogram, 100, 80).duration(1.0);
        stats.record(&features);

        assert!((stats.frames_per_second() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_stats_format() {
        let mut stats = AudioStats::new();
        let features = AudioFeatures::new(FeatureType::MelSpectrogram, 100, 80).duration(10.0);
        stats.record(&features);

        let formatted = stats.format();
        assert!(formatted.contains("10.0s total"));
        assert!(formatted.contains("100 frames"));
        assert!(formatted.contains("1 files"));
    }

    #[test]
    fn test_hz_to_mel() {
        let mel = hz_to_mel(1000.0);
        assert!(mel > 0.0);
        assert!((mel - 1000.0).abs() < 100.0); // Rough check
    }

    #[test]
    fn test_mel_to_hz() {
        let hz = mel_to_hz(1000.0);
        assert!(hz > 0.0);
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        let original_hz = 440.0;
        let mel = hz_to_mel(original_hz);
        let recovered_hz = mel_to_hz(mel);
        assert!((original_hz - recovered_hz).abs() < 0.1);
    }

    #[test]
    fn test_calculate_num_frames() {
        assert_eq!(calculate_num_frames(16000, 400, 160), 98);
        assert_eq!(calculate_num_frames(400, 400, 160), 1);
        assert_eq!(calculate_num_frames(200, 400, 160), 1);
    }

    #[test]
    fn test_estimate_extraction_time() {
        let waveform_time = estimate_extraction_time(10.0, FeatureType::Waveform);
        let mfcc_time = estimate_extraction_time(10.0, FeatureType::Mfcc);

        assert!(waveform_time > 0);
        assert!(mfcc_time > waveform_time); // MFCC takes longer
    }
}
