//! Speech Recognition
//!
//! ASR models and speech processing pipelines.
//!
//! # Python Equivalent
//! ```python
//! from hf_gtc.audio import SpeechModel, TranscriptionConfig, transcribe
//! result = transcribe(audio, model=SpeechModel.WHISPER_SMALL)
//! ```

/// Speech recognition model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeechModel {
    /// Whisper tiny (39M params)
    WhisperTiny,
    /// Whisper base (74M params)
    WhisperBase,
    /// Whisper small (244M params, default)
    #[default]
    WhisperSmall,
    /// Whisper medium (769M params)
    WhisperMedium,
    /// Whisper large (1.5B params)
    WhisperLarge,
    /// Wav2Vec2 base
    Wav2Vec2Base,
    /// Wav2Vec2 large
    Wav2Vec2Large,
}

impl SpeechModel {
    /// Get model name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WhisperTiny => "whisper_tiny",
            Self::WhisperBase => "whisper_base",
            Self::WhisperSmall => "whisper_small",
            Self::WhisperMedium => "whisper_medium",
            Self::WhisperLarge => "whisper_large",
            Self::Wav2Vec2Base => "wav2vec2_base",
            Self::Wav2Vec2Large => "wav2vec2_large",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "whisper_tiny" | "tiny" => Some(Self::WhisperTiny),
            "whisper_base" | "base" => Some(Self::WhisperBase),
            "whisper_small" | "small" => Some(Self::WhisperSmall),
            "whisper_medium" | "medium" => Some(Self::WhisperMedium),
            "whisper_large" | "large" => Some(Self::WhisperLarge),
            "wav2vec2_base" | "w2v_base" => Some(Self::Wav2Vec2Base),
            "wav2vec2_large" | "w2v_large" => Some(Self::Wav2Vec2Large),
            _ => None,
        }
    }

    /// List all models
    pub fn list_all() -> Vec<Self> {
        vec![
            Self::WhisperTiny,
            Self::WhisperBase,
            Self::WhisperSmall,
            Self::WhisperMedium,
            Self::WhisperLarge,
            Self::Wav2Vec2Base,
            Self::Wav2Vec2Large,
        ]
    }

    /// Get parameter count (millions)
    pub fn param_count_millions(&self) -> usize {
        match self {
            Self::WhisperTiny => 39,
            Self::WhisperBase => 74,
            Self::WhisperSmall => 244,
            Self::WhisperMedium => 769,
            Self::WhisperLarge => 1550,
            Self::Wav2Vec2Base => 95,
            Self::Wav2Vec2Large => 317,
        }
    }

    /// Check if Whisper model
    pub fn is_whisper(&self) -> bool {
        matches!(
            self,
            Self::WhisperTiny
                | Self::WhisperBase
                | Self::WhisperSmall
                | Self::WhisperMedium
                | Self::WhisperLarge
        )
    }

    /// Get encoder dimension
    pub fn encoder_dim(&self) -> usize {
        match self {
            Self::WhisperTiny => 384,
            Self::WhisperBase => 512,
            Self::WhisperSmall => 768,
            Self::WhisperMedium => 1024,
            Self::WhisperLarge => 1280,
            Self::Wav2Vec2Base => 768,
            Self::Wav2Vec2Large => 1024,
        }
    }
}

/// Task type for speech models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeechTask {
    /// Automatic speech recognition (default)
    #[default]
    Transcribe,
    /// Translation to English
    Translate,
    /// Language identification
    LanguageId,
    /// Voice activity detection
    Vad,
    /// Speaker diarization
    Diarization,
}

impl SpeechTask {
    /// Get task name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
            Self::LanguageId => "language_id",
            Self::Vad => "vad",
            Self::Diarization => "diarization",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "transcribe" | "asr" => Some(Self::Transcribe),
            "translate" | "translation" => Some(Self::Translate),
            "language_id" | "lid" | "language" => Some(Self::LanguageId),
            "vad" | "voice_activity" => Some(Self::Vad),
            "diarization" | "diarize" => Some(Self::Diarization),
            _ => None,
        }
    }
}

/// Decoding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodingStrategy {
    /// Greedy decoding (default)
    #[default]
    Greedy,
    /// Beam search
    BeamSearch,
    /// Best of N sampling
    BestOfN,
}

impl DecodingStrategy {
    /// Get strategy name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::BeamSearch => "beam_search",
            Self::BestOfN => "best_of_n",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "greedy" => Some(Self::Greedy),
            "beam_search" | "beam" => Some(Self::BeamSearch),
            "best_of_n" | "sampling" => Some(Self::BestOfN),
            _ => None,
        }
    }
}

/// Speech recognition configuration
#[derive(Debug, Clone)]
pub struct SpeechConfig {
    /// Model to use
    pub model: SpeechModel,
    /// Task type
    pub task: SpeechTask,
    /// Decoding strategy
    pub decoding: DecodingStrategy,
    /// Language code (ISO 639-1)
    pub language: Option<String>,
    /// Beam size for beam search
    pub beam_size: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Enable timestamps
    pub timestamps: bool,
    /// Word-level timestamps
    pub word_timestamps: bool,
    /// Initial prompt
    pub initial_prompt: Option<String>,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            model: SpeechModel::WhisperSmall,
            task: SpeechTask::Transcribe,
            decoding: DecodingStrategy::Greedy,
            language: None,
            beam_size: 5,
            temperature: 0.0,
            timestamps: false,
            word_timestamps: false,
            initial_prompt: None,
        }
    }
}

impl SpeechConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model
    pub fn model(mut self, m: SpeechModel) -> Self {
        self.model = m;
        self
    }

    /// Set task
    pub fn task(mut self, t: SpeechTask) -> Self {
        self.task = t;
        self
    }

    /// Set decoding strategy
    pub fn decoding(mut self, d: DecodingStrategy) -> Self {
        self.decoding = d;
        self
    }

    /// Set language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set beam size
    pub fn beam_size(mut self, size: usize) -> Self {
        self.beam_size = size;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.max(0.0);
        self
    }

    /// Enable timestamps
    pub fn timestamps(mut self, enabled: bool) -> Self {
        self.timestamps = enabled;
        self
    }

    /// Enable word timestamps
    pub fn word_timestamps(mut self, enabled: bool) -> Self {
        self.word_timestamps = enabled;
        if enabled {
            self.timestamps = true;
        }
        self
    }

    /// Set initial prompt
    pub fn initial_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.initial_prompt = Some(prompt.into());
        self
    }

    /// Validate config
    pub fn validate(&self) -> Result<(), String> {
        if self.beam_size == 0 {
            return Err("Beam size must be > 0".to_string());
        }
        if self.decoding == DecodingStrategy::BeamSearch && self.beam_size < 2 {
            return Err("Beam search requires beam_size >= 2".to_string());
        }
        if self.task == SpeechTask::Translate && !self.model.is_whisper() {
            return Err("Translation only supported for Whisper models".to_string());
        }
        Ok(())
    }
}

/// Transcription segment
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Segment text
    pub text: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence score
    pub confidence: f32,
    /// Speaker ID (for diarization)
    pub speaker: Option<String>,
}

impl TranscriptionSegment {
    /// Create new segment
    pub fn new(text: impl Into<String>, start: f32, end: f32) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            confidence: 1.0,
            speaker: None,
        }
    }

    /// Set confidence
    pub fn confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    /// Set speaker
    pub fn speaker(mut self, s: impl Into<String>) -> Self {
        self.speaker = Some(s.into());
        self
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Get words per second (approximate)
    pub fn words_per_second(&self) -> f32 {
        let word_count = self.text.split_whitespace().count();
        let duration = self.duration();
        if duration > 0.0 {
            word_count as f32 / duration
        } else {
            0.0
        }
    }
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Full transcribed text
    pub text: String,
    /// Segments with timestamps
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language
    pub language: Option<String>,
    /// Language confidence
    pub language_confidence: f32,
    /// Total duration processed
    pub duration_secs: f32,
    /// Processing time
    pub processing_time_ms: u64,
}

impl TranscriptionResult {
    /// Create new result
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            segments: Vec::new(),
            language: None,
            language_confidence: 0.0,
            duration_secs: 0.0,
            processing_time_ms: 0,
        }
    }

    /// Add segment
    pub fn add_segment(mut self, segment: TranscriptionSegment) -> Self {
        self.segments.push(segment);
        self
    }

    /// Set language
    pub fn language(mut self, lang: impl Into<String>, confidence: f32) -> Self {
        self.language = Some(lang.into());
        self.language_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set duration
    pub fn duration(mut self, secs: f32) -> Self {
        self.duration_secs = secs;
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

    /// Get character count
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Get average confidence across segments
    pub fn average_confidence(&self) -> f32 {
        if self.segments.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.segments.iter().map(|s| s.confidence).sum();
        sum / self.segments.len() as f32
    }

    /// Get real-time factor (processing time / audio duration)
    pub fn real_time_factor(&self) -> f32 {
        if self.duration_secs == 0.0 {
            return 0.0;
        }
        (self.processing_time_ms as f32 / 1000.0) / self.duration_secs
    }
}

/// Speech recognition statistics
#[derive(Debug, Clone, Default)]
pub struct SpeechStats {
    /// Total audio transcribed (seconds)
    pub total_audio_secs: f32,
    /// Total processing time (ms)
    pub total_processing_ms: u64,
    /// Total words transcribed
    pub total_words: usize,
    /// Files processed
    pub files_processed: usize,
}

impl SpeechStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record transcription
    pub fn record(&mut self, result: &TranscriptionResult) {
        self.total_audio_secs += result.duration_secs;
        self.total_processing_ms += result.processing_time_ms;
        self.total_words += result.word_count();
        self.files_processed += 1;
    }

    /// Get overall real-time factor
    pub fn real_time_factor(&self) -> f32 {
        if self.total_audio_secs == 0.0 {
            return 0.0;
        }
        (self.total_processing_ms as f32 / 1000.0) / self.total_audio_secs
    }

    /// Get words per second
    pub fn words_per_second(&self) -> f32 {
        if self.total_audio_secs == 0.0 {
            return 0.0;
        }
        self.total_words as f32 / self.total_audio_secs
    }

    /// Format as string
    pub fn format(&self) -> String {
        format!(
            "Speech: {:.1}s audio, {} words, RTF={:.2}x",
            self.total_audio_secs,
            self.total_words,
            self.real_time_factor()
        )
    }
}

/// Estimate transcription time in ms
pub fn estimate_transcription_time(duration_secs: f32, model: SpeechModel) -> u64 {
    let base_rtf = match model {
        SpeechModel::WhisperTiny => 0.1,
        SpeechModel::WhisperBase => 0.2,
        SpeechModel::WhisperSmall => 0.5,
        SpeechModel::WhisperMedium => 1.0,
        SpeechModel::WhisperLarge => 2.0,
        SpeechModel::Wav2Vec2Base => 0.3,
        SpeechModel::Wav2Vec2Large => 0.6,
    };
    (duration_secs * base_rtf * 1000.0) as u64
}

/// Estimate memory usage in MB
pub fn estimate_memory_mb(model: SpeechModel, batch_size: usize) -> usize {
    let base_memory = model.param_count_millions() * 4; // 4 bytes per param
    let activation_memory = model.encoder_dim() * 1500 * batch_size / 1024; // Rough estimate
    base_memory + activation_memory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_model_default() {
        assert_eq!(SpeechModel::default(), SpeechModel::WhisperSmall);
    }

    #[test]
    fn test_speech_model_as_str() {
        assert_eq!(SpeechModel::WhisperTiny.as_str(), "whisper_tiny");
        assert_eq!(SpeechModel::WhisperBase.as_str(), "whisper_base");
        assert_eq!(SpeechModel::WhisperSmall.as_str(), "whisper_small");
        assert_eq!(SpeechModel::WhisperMedium.as_str(), "whisper_medium");
        assert_eq!(SpeechModel::WhisperLarge.as_str(), "whisper_large");
        assert_eq!(SpeechModel::Wav2Vec2Base.as_str(), "wav2vec2_base");
        assert_eq!(SpeechModel::Wav2Vec2Large.as_str(), "wav2vec2_large");
    }

    #[test]
    fn test_speech_model_parse() {
        assert_eq!(SpeechModel::parse("tiny"), Some(SpeechModel::WhisperTiny));
        assert_eq!(SpeechModel::parse("base"), Some(SpeechModel::WhisperBase));
        assert_eq!(SpeechModel::parse("small"), Some(SpeechModel::WhisperSmall));
        assert_eq!(SpeechModel::parse("medium"), Some(SpeechModel::WhisperMedium));
        assert_eq!(SpeechModel::parse("large"), Some(SpeechModel::WhisperLarge));
        assert_eq!(SpeechModel::parse("w2v_base"), Some(SpeechModel::Wav2Vec2Base));
        assert_eq!(SpeechModel::parse("w2v_large"), Some(SpeechModel::Wav2Vec2Large));
        assert_eq!(SpeechModel::parse("unknown"), None);
    }

    #[test]
    fn test_speech_model_list_all() {
        assert_eq!(SpeechModel::list_all().len(), 7);
    }

    #[test]
    fn test_speech_model_param_count() {
        assert_eq!(SpeechModel::WhisperTiny.param_count_millions(), 39);
        assert_eq!(SpeechModel::WhisperLarge.param_count_millions(), 1550);
    }

    #[test]
    fn test_speech_model_is_whisper() {
        assert!(SpeechModel::WhisperSmall.is_whisper());
        assert!(!SpeechModel::Wav2Vec2Base.is_whisper());
    }

    #[test]
    fn test_speech_model_encoder_dim() {
        assert_eq!(SpeechModel::WhisperTiny.encoder_dim(), 384);
        assert_eq!(SpeechModel::WhisperSmall.encoder_dim(), 768);
        assert_eq!(SpeechModel::WhisperLarge.encoder_dim(), 1280);
    }

    #[test]
    fn test_speech_task_as_str() {
        assert_eq!(SpeechTask::Transcribe.as_str(), "transcribe");
        assert_eq!(SpeechTask::Translate.as_str(), "translate");
        assert_eq!(SpeechTask::LanguageId.as_str(), "language_id");
        assert_eq!(SpeechTask::Vad.as_str(), "vad");
        assert_eq!(SpeechTask::Diarization.as_str(), "diarization");
    }

    #[test]
    fn test_speech_task_parse() {
        assert_eq!(SpeechTask::parse("asr"), Some(SpeechTask::Transcribe));
        assert_eq!(SpeechTask::parse("translation"), Some(SpeechTask::Translate));
        assert_eq!(SpeechTask::parse("lid"), Some(SpeechTask::LanguageId));
        assert_eq!(SpeechTask::parse("voice_activity"), Some(SpeechTask::Vad));
        assert_eq!(SpeechTask::parse("diarize"), Some(SpeechTask::Diarization));
        assert_eq!(SpeechTask::parse("unknown"), None);
    }

    #[test]
    fn test_decoding_strategy_as_str() {
        assert_eq!(DecodingStrategy::Greedy.as_str(), "greedy");
        assert_eq!(DecodingStrategy::BeamSearch.as_str(), "beam_search");
        assert_eq!(DecodingStrategy::BestOfN.as_str(), "best_of_n");
    }

    #[test]
    fn test_decoding_strategy_parse() {
        assert_eq!(DecodingStrategy::parse("greedy"), Some(DecodingStrategy::Greedy));
        assert_eq!(DecodingStrategy::parse("beam"), Some(DecodingStrategy::BeamSearch));
        assert_eq!(DecodingStrategy::parse("sampling"), Some(DecodingStrategy::BestOfN));
        assert_eq!(DecodingStrategy::parse("unknown"), None);
    }

    #[test]
    fn test_speech_config_default() {
        let config = SpeechConfig::default();
        assert_eq!(config.model, SpeechModel::WhisperSmall);
        assert_eq!(config.task, SpeechTask::Transcribe);
        assert_eq!(config.decoding, DecodingStrategy::Greedy);
    }

    #[test]
    fn test_speech_config_builder() {
        let config = SpeechConfig::new()
            .model(SpeechModel::WhisperMedium)
            .task(SpeechTask::Translate)
            .decoding(DecodingStrategy::BeamSearch)
            .language("en")
            .beam_size(10)
            .temperature(0.2)
            .timestamps(true)
            .word_timestamps(true)
            .initial_prompt("Hello");

        assert_eq!(config.model, SpeechModel::WhisperMedium);
        assert_eq!(config.task, SpeechTask::Translate);
        assert_eq!(config.decoding, DecodingStrategy::BeamSearch);
        assert_eq!(config.language, Some("en".to_string()));
        assert_eq!(config.beam_size, 10);
        assert!((config.temperature - 0.2).abs() < 0.001);
        assert!(config.timestamps);
        assert!(config.word_timestamps);
        assert_eq!(config.initial_prompt, Some("Hello".to_string()));
    }

    #[test]
    fn test_speech_config_validate() {
        let valid = SpeechConfig::default();
        assert!(valid.validate().is_ok());

        let zero_beam = SpeechConfig::new().beam_size(0);
        assert!(zero_beam.validate().is_err());

        let bad_beam_search = SpeechConfig::new()
            .decoding(DecodingStrategy::BeamSearch)
            .beam_size(1);
        assert!(bad_beam_search.validate().is_err());

        let bad_translate = SpeechConfig::new()
            .model(SpeechModel::Wav2Vec2Base)
            .task(SpeechTask::Translate);
        assert!(bad_translate.validate().is_err());
    }

    #[test]
    fn test_transcription_segment_new() {
        let segment = TranscriptionSegment::new("Hello world", 0.0, 2.0);
        assert_eq!(segment.text, "Hello world");
        assert_eq!(segment.start, 0.0);
        assert_eq!(segment.end, 2.0);
    }

    #[test]
    fn test_transcription_segment_builder() {
        let segment = TranscriptionSegment::new("Test", 1.0, 3.0)
            .confidence(0.95)
            .speaker("speaker_1");

        assert!((segment.confidence - 0.95).abs() < 0.001);
        assert_eq!(segment.speaker, Some("speaker_1".to_string()));
    }

    #[test]
    fn test_transcription_segment_duration() {
        let segment = TranscriptionSegment::new("Test", 1.5, 4.5);
        assert!((segment.duration() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_transcription_segment_words_per_second() {
        let segment = TranscriptionSegment::new("Hello world test", 0.0, 3.0);
        let wps = segment.words_per_second();
        assert!((wps - 1.0).abs() < 0.001); // 3 words in 3 seconds
    }

    #[test]
    fn test_transcription_result_new() {
        let result = TranscriptionResult::new("Hello world");
        assert_eq!(result.text, "Hello world");
        assert!(result.segments.is_empty());
    }

    #[test]
    fn test_transcription_result_builder() {
        let result = TranscriptionResult::new("Hello world")
            .add_segment(TranscriptionSegment::new("Hello", 0.0, 0.5))
            .add_segment(TranscriptionSegment::new("world", 0.5, 1.0))
            .language("en", 0.99)
            .duration(1.0)
            .processing_time(500);

        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.language, Some("en".to_string()));
        assert!((result.language_confidence - 0.99).abs() < 0.001);
        assert!((result.duration_secs - 1.0).abs() < 0.001);
        assert_eq!(result.processing_time_ms, 500);
    }

    #[test]
    fn test_transcription_result_word_count() {
        let result = TranscriptionResult::new("Hello world test");
        assert_eq!(result.word_count(), 3);
    }

    #[test]
    fn test_transcription_result_char_count() {
        let result = TranscriptionResult::new("Hello");
        assert_eq!(result.char_count(), 5);
    }

    #[test]
    fn test_transcription_result_average_confidence() {
        let result = TranscriptionResult::new("Test")
            .add_segment(TranscriptionSegment::new("A", 0.0, 0.5).confidence(0.9))
            .add_segment(TranscriptionSegment::new("B", 0.5, 1.0).confidence(0.8));

        assert!((result.average_confidence() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_transcription_result_real_time_factor() {
        let result = TranscriptionResult::new("Test")
            .duration(10.0)
            .processing_time(5000); // 5 seconds

        assert!((result.real_time_factor() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_speech_stats_new() {
        let stats = SpeechStats::new();
        assert_eq!(stats.files_processed, 0);
        assert_eq!(stats.total_words, 0);
    }

    #[test]
    fn test_speech_stats_record() {
        let mut stats = SpeechStats::new();
        let result = TranscriptionResult::new("Hello world test")
            .duration(3.0)
            .processing_time(1500);

        stats.record(&result);

        assert_eq!(stats.files_processed, 1);
        assert_eq!(stats.total_words, 3);
        assert!((stats.total_audio_secs - 3.0).abs() < 0.001);
        assert_eq!(stats.total_processing_ms, 1500);
    }

    #[test]
    fn test_speech_stats_real_time_factor() {
        let mut stats = SpeechStats::new();
        let result = TranscriptionResult::new("Test")
            .duration(10.0)
            .processing_time(5000);
        stats.record(&result);

        assert!((stats.real_time_factor() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_speech_stats_words_per_second() {
        let mut stats = SpeechStats::new();
        let result = TranscriptionResult::new("one two three four five")
            .duration(5.0);
        stats.record(&result);

        assert!((stats.words_per_second() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speech_stats_format() {
        let mut stats = SpeechStats::new();
        let result = TranscriptionResult::new("Hello world")
            .duration(2.0)
            .processing_time(1000);
        stats.record(&result);

        let formatted = stats.format();
        assert!(formatted.contains("2.0s audio"));
        assert!(formatted.contains("2 words"));
    }

    #[test]
    fn test_estimate_transcription_time() {
        let tiny_time = estimate_transcription_time(60.0, SpeechModel::WhisperTiny);
        let large_time = estimate_transcription_time(60.0, SpeechModel::WhisperLarge);

        assert!(tiny_time > 0);
        assert!(large_time > tiny_time); // Larger model takes longer
    }

    #[test]
    fn test_estimate_memory_mb() {
        let tiny_memory = estimate_memory_mb(SpeechModel::WhisperTiny, 1);
        let large_memory = estimate_memory_mb(SpeechModel::WhisperLarge, 1);

        assert!(tiny_memory > 0);
        assert!(large_memory > tiny_memory);
    }
}
