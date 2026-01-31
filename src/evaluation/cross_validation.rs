//! Cross-validation utilities

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_state: Option<u64>,
    /// Stratified splitting (for classification)
    pub stratified: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            shuffle: true,
            random_state: None,
            stratified: false,
        }
    }
}

impl CrossValidationConfig {
    /// Set number of folds
    pub fn n_folds(mut self, n: usize) -> Self {
        self.n_folds = n;
        self
    }

    /// Enable/disable shuffling
    pub fn shuffle(mut self, enable: bool) -> Self {
        self.shuffle = enable;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable/disable stratified splitting
    pub fn stratified(mut self, enable: bool) -> Self {
        self.stratified = enable;
        self
    }
}

/// Result from one fold
#[derive(Debug, Clone, Default)]
pub struct FoldResult {
    /// Fold index
    pub fold: usize,
    /// Training score
    pub train_score: f64,
    /// Validation score
    pub val_score: f64,
    /// Training indices
    pub train_indices: Vec<usize>,
    /// Validation indices
    pub val_indices: Vec<usize>,
}

impl FoldResult {
    /// Create new fold result
    pub fn new(fold: usize, train_score: f64, val_score: f64) -> Self {
        Self {
            fold,
            train_score,
            val_score,
            train_indices: Vec::new(),
            val_indices: Vec::new(),
        }
    }

    /// Set indices
    pub fn with_indices(mut self, train: Vec<usize>, val: Vec<usize>) -> Self {
        self.train_indices = train;
        self.val_indices = val;
        self
    }
}

/// Cross-validator for model evaluation
#[derive(Debug)]
pub struct CrossValidator {
    config: CrossValidationConfig,
    results: Vec<FoldResult>,
}

impl CrossValidator {
    /// Create new cross-validator
    pub fn new(config: CrossValidationConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CrossValidationConfig {
        &self.config
    }

    /// Get number of folds
    pub fn n_folds(&self) -> usize {
        self.config.n_folds
    }

    /// Generate fold indices for k-fold cross-validation
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        if n_samples == 0 || self.config.n_folds == 0 {
            return Vec::new();
        }

        let indices: Vec<usize> = if self.config.shuffle {
            self.shuffled_indices(n_samples)
        } else {
            (0..n_samples).collect()
        };

        let fold_size = n_samples / self.config.n_folds;
        let remainder = n_samples % self.config.n_folds;

        let mut folds = Vec::with_capacity(self.config.n_folds);
        let mut start = 0;

        for i in 0..self.config.n_folds {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + fold_size + extra;

            let val_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            folds.push((train_indices, val_indices));
            start = end;
        }

        folds
    }

    fn shuffled_indices(&self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();

        // Simple LCG shuffle if random_state provided
        if let Some(seed) = self.config.random_state {
            let mut rng = seed;
            for i in (1..n).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        indices
    }

    /// Record a fold result
    pub fn record_fold(&mut self, result: FoldResult) {
        self.results.push(result);
    }

    /// Get all results
    pub fn results(&self) -> &[FoldResult] {
        &self.results
    }

    /// Calculate mean validation score
    pub fn mean_val_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.val_score).sum();
        sum / self.results.len() as f64
    }

    /// Calculate standard deviation of validation scores
    pub fn std_val_score(&self) -> f64 {
        if self.results.len() < 2 {
            return 0.0;
        }

        let mean = self.mean_val_score();
        let variance: f64 = self.results
            .iter()
            .map(|r| (r.val_score - mean).powi(2))
            .sum::<f64>() / (self.results.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate mean training score
    pub fn mean_train_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.train_score).sum();
        sum / self.results.len() as f64
    }

    /// Reset results
    pub fn reset(&mut self) {
        self.results.clear();
    }
}

impl Default for CrossValidator {
    fn default() -> Self {
        Self::new(CrossValidationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_config_default() {
        let config = CrossValidationConfig::default();
        assert_eq!(config.n_folds, 5);
        assert!(config.shuffle);
        assert!(config.random_state.is_none());
        assert!(!config.stratified);
    }

    #[test]
    fn test_cv_config_builder() {
        let config = CrossValidationConfig::default()
            .n_folds(10)
            .shuffle(false)
            .random_state(42)
            .stratified(true);

        assert_eq!(config.n_folds, 10);
        assert!(!config.shuffle);
        assert_eq!(config.random_state, Some(42));
        assert!(config.stratified);
    }

    #[test]
    fn test_fold_result_new() {
        let result = FoldResult::new(0, 0.9, 0.85);
        assert_eq!(result.fold, 0);
        assert!((result.train_score - 0.9).abs() < 1e-10);
        assert!((result.val_score - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_fold_result_with_indices() {
        let result = FoldResult::new(0, 0.9, 0.85)
            .with_indices(vec![0, 1, 2], vec![3, 4]);

        assert_eq!(result.train_indices, vec![0, 1, 2]);
        assert_eq!(result.val_indices, vec![3, 4]);
    }

    #[test]
    fn test_cross_validator_new() {
        let cv = CrossValidator::default();
        assert_eq!(cv.n_folds(), 5);
        assert!(cv.results().is_empty());
    }

    #[test]
    fn test_cross_validator_split() {
        let config = CrossValidationConfig::default().n_folds(5).shuffle(false);
        let cv = CrossValidator::new(config);

        let folds = cv.split(10);

        assert_eq!(folds.len(), 5);

        // Each fold should have 2 validation samples
        for (train, val) in &folds {
            assert_eq!(val.len(), 2);
            assert_eq!(train.len(), 8);
        }

        // All indices should be covered
        let all_val: Vec<usize> = folds.iter().flat_map(|(_, v)| v.clone()).collect();
        let mut sorted = all_val.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_cross_validator_split_uneven() {
        let config = CrossValidationConfig::default().n_folds(3).shuffle(false);
        let cv = CrossValidator::new(config);

        let folds = cv.split(10);

        assert_eq!(folds.len(), 3);

        // Total validation samples should equal total samples
        let total: usize = folds.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_cross_validator_split_empty() {
        let cv = CrossValidator::default();
        let folds = cv.split(0);
        assert!(folds.is_empty());
    }

    #[test]
    fn test_cross_validator_split_zero_folds() {
        let config = CrossValidationConfig::default().n_folds(0);
        let cv = CrossValidator::new(config);
        let folds = cv.split(10);
        assert!(folds.is_empty());
    }

    #[test]
    fn test_cross_validator_split_shuffle() {
        let config = CrossValidationConfig::default()
            .n_folds(2)
            .shuffle(true)
            .random_state(42);
        let cv = CrossValidator::new(config);

        let folds = cv.split(10);
        let (train1, val1) = &folds[0];

        // Check that indices are shuffled (not sequential)
        let is_sequential = val1.windows(2).all(|w| w[1] == w[0] + 1);
        // With shuffling, unlikely to be perfectly sequential
        assert!(!is_sequential || val1.len() <= 1);

        // Still should cover all indices
        let all: Vec<usize> = folds.iter().flat_map(|(_, v)| v.clone()).collect();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_cross_validator_record_fold() {
        let mut cv = CrossValidator::default();

        cv.record_fold(FoldResult::new(0, 0.9, 0.85));
        cv.record_fold(FoldResult::new(1, 0.92, 0.87));

        assert_eq!(cv.results().len(), 2);
    }

    #[test]
    fn test_cross_validator_mean_val_score() {
        let mut cv = CrossValidator::default();

        cv.record_fold(FoldResult::new(0, 0.9, 0.80));
        cv.record_fold(FoldResult::new(1, 0.9, 0.85));
        cv.record_fold(FoldResult::new(2, 0.9, 0.90));

        assert!((cv.mean_val_score() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validator_mean_val_score_empty() {
        let cv = CrossValidator::default();
        assert!((cv.mean_val_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validator_std_val_score() {
        let mut cv = CrossValidator::default();

        cv.record_fold(FoldResult::new(0, 0.9, 0.80));
        cv.record_fold(FoldResult::new(1, 0.9, 0.90));

        // std = sqrt(((0.80-0.85)^2 + (0.90-0.85)^2) / 1) = sqrt(0.005) = 0.0707...
        assert!((cv.std_val_score() - 0.0707106781).abs() < 1e-6);
    }

    #[test]
    fn test_cross_validator_std_val_score_single() {
        let mut cv = CrossValidator::default();
        cv.record_fold(FoldResult::new(0, 0.9, 0.85));
        assert!((cv.std_val_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validator_std_val_score_empty() {
        let cv = CrossValidator::default();
        assert!((cv.std_val_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validator_mean_train_score() {
        let mut cv = CrossValidator::default();

        cv.record_fold(FoldResult::new(0, 0.80, 0.70));
        cv.record_fold(FoldResult::new(1, 0.90, 0.80));

        assert!((cv.mean_train_score() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validator_mean_train_score_empty() {
        let cv = CrossValidator::default();
        assert!((cv.mean_train_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validator_reset() {
        let mut cv = CrossValidator::default();
        cv.record_fold(FoldResult::new(0, 0.9, 0.85));
        cv.reset();
        assert!(cv.results().is_empty());
    }

    #[test]
    fn test_cross_validator_config_access() {
        let config = CrossValidationConfig::default().n_folds(10);
        let cv = CrossValidator::new(config);
        assert_eq!(cv.config().n_folds, 10);
    }

    #[test]
    fn test_fold_result_default() {
        let result = FoldResult::default();
        assert_eq!(result.fold, 0);
        assert!((result.train_score - 0.0).abs() < 1e-10);
        assert!((result.val_score - 0.0).abs() < 1e-10);
        assert!(result.train_indices.is_empty());
        assert!(result.val_indices.is_empty());
    }

    #[test]
    fn test_split_deterministic() {
        let config = CrossValidationConfig::default()
            .n_folds(3)
            .shuffle(true)
            .random_state(42);

        let cv1 = CrossValidator::new(config.clone());
        let cv2 = CrossValidator::new(config);

        let folds1 = cv1.split(10);
        let folds2 = cv2.split(10);

        // Same seed should produce same splits
        assert_eq!(folds1, folds2);
    }
}
