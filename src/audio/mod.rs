//! Audio Module
//!
//! Speech recognition, audio features, and transcription.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.audio import (
//!     FeatureType, AudioConfig, extract_features,
//!     SpeechModel, SpeechConfig, transcribe,
//! )
//! ```
//!
//! # Submodules
//! - `features`: Audio feature extraction (mel spectrograms, MFCCs)
//! - `speech`: Speech recognition and transcription

pub mod features;
pub mod speech;

// Re-export key types
pub use features::{
    AudioConfig, AudioFeatures, AudioStats, FeatureType, WindowType,
};
pub use speech::{
    DecodingStrategy, SpeechConfig, SpeechModel, SpeechStats, SpeechTask,
    TranscriptionResult, TranscriptionSegment,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_re_exports() {
        let _ = FeatureType::default();
        let _ = AudioConfig::default();
        let _ = WindowType::default();
    }

    #[test]
    fn test_speech_re_exports() {
        let _ = SpeechModel::default();
        let _ = SpeechConfig::default();
        let _ = SpeechTask::default();
        let _ = DecodingStrategy::default();
    }

    #[test]
    fn test_integration_audio_pipeline() {
        // Feature extraction config
        let audio_config = AudioConfig::whisper()
            .n_mels(80)
            .sample_rate(16000);

        // Speech recognition config
        let speech_config = SpeechConfig::new()
            .model(SpeechModel::WhisperSmall)
            .task(SpeechTask::Transcribe)
            .timestamps(true);

        assert_eq!(audio_config.n_mels, 80);
        assert_eq!(speech_config.model, SpeechModel::WhisperSmall);
        assert!(speech_config.timestamps);
    }

    #[test]
    fn test_integration_transcription_with_features() {
        let features = AudioFeatures::new(FeatureType::LogMel, 100, 80)
            .duration(10.0)
            .sample_rate(16000);

        let result = TranscriptionResult::new("Hello world")
            .duration(features.duration_secs)
            .add_segment(TranscriptionSegment::new("Hello", 0.0, 0.5))
            .add_segment(TranscriptionSegment::new("world", 0.5, 1.0));

        assert_eq!(features.num_frames, 100);
        assert_eq!(result.segments.len(), 2);
        assert!((result.duration_secs - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_preset_configs() {
        let whisper_config = AudioConfig::whisper();
        let wav2vec_config = AudioConfig::wav2vec();

        assert_eq!(whisper_config.feature_type, FeatureType::LogMel);
        assert_eq!(wav2vec_config.feature_type, FeatureType::Waveform);
    }
}
