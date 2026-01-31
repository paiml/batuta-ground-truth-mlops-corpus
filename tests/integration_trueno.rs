//! Integration Tests with Trueno SIMD Operations
//!
//! These tests verify that the corpus types integrate correctly
//! with trueno matrix and vector operations.

use batuta_ground_truth_mlops_corpus::preprocessing::{Tokenizer, TokenizerConfig};
use batuta_ground_truth_mlops_corpus::models::{RandomForestConfig, TransformerConfig};
use batuta_ground_truth_mlops_corpus::training::{TrainerConfig, LearningRateScheduler};
use batuta_ground_truth_mlops_corpus::inference::{BatchConfig, PipelineConfig};
use batuta_ground_truth_mlops_corpus::evaluation::{ClassificationMetrics, CrossValidationConfig};
use batuta_ground_truth_mlops_corpus::deployment::{QuantizationConfig, QuantizationType};

use trueno::{Matrix, Vector};

// =============================================================================
// Preprocessing + Trueno Integration
// =============================================================================

#[test]
fn test_tokenizer_with_trueno_vector() {
    let tokenizer = Tokenizer::new(TokenizerConfig::default());
    let tokens = tokenizer.tokenize("Hello world from trueno");

    // Convert token lengths to trueno vector
    let token_lens: Vec<f32> = tokens.iter().map(|t| t.text.len() as f32).collect();
    let vector = Vector::from_slice(&token_lens);

    assert_eq!(vector.len(), token_lens.len());
    assert!(!tokens.is_empty());
}

#[test]
fn test_token_embeddings_matrix() {
    let tokenizer = Tokenizer::new(TokenizerConfig::default());
    let tokens = tokenizer.tokenize("Machine learning with Rust");

    // Simulate embeddings as 2D matrix [num_tokens, embed_dim]
    let embed_dim = 64;
    let num_tokens = tokens.len();
    let embeddings: Vec<f32> = (0..num_tokens * embed_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let matrix = Matrix::from_vec(num_tokens, embed_dim, embeddings).unwrap();
    assert_eq!(matrix.rows(), num_tokens);
    assert_eq!(matrix.cols(), embed_dim);
}

// =============================================================================
// Model Config + Trueno Integration
// =============================================================================

#[test]
fn test_random_forest_weights_matrix() {
    let config = RandomForestConfig::default()
        .n_estimators(10)
        .max_depth(Some(5));

    // Simulate tree weights as matrix
    let num_trees = config.n_estimators;
    let weights_per_tree = 32;
    let weights: Vec<f32> = (0..num_trees * weights_per_tree)
        .map(|i| ((i % 100) as f32) / 100.0)
        .collect();

    let matrix = Matrix::from_vec(num_trees, weights_per_tree, weights).unwrap();
    assert_eq!(matrix.rows(), num_trees);
    assert_eq!(matrix.cols(), weights_per_tree);
}

#[test]
fn test_transformer_attention_matrix() {
    let config = TransformerConfig::bert_base();

    // Simulate attention weights [seq_len, seq_len] for one head
    let seq_len = 16;
    let num_heads = config.num_attention_heads;

    let attention: Vec<f32> = (0..seq_len * seq_len)
        .map(|_| 1.0 / (seq_len as f32))
        .collect();

    let matrix = Matrix::from_vec(seq_len, seq_len, attention).unwrap();
    assert_eq!(matrix.rows(), seq_len);
    assert_eq!(matrix.cols(), seq_len);
    assert_eq!(num_heads, 12);
}

#[test]
fn test_transformer_hidden_states() {
    let config = TransformerConfig::bert_large();

    // Simulate hidden states [seq_len, hidden_size]
    let seq_len = 32;
    let hidden_size = config.hidden_size;

    let hidden: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (i as f32).sin())
        .collect();

    let matrix = Matrix::from_vec(seq_len, hidden_size, hidden).unwrap();
    assert_eq!(matrix.rows(), seq_len);
    assert_eq!(matrix.cols(), hidden_size);
}

// =============================================================================
// Training + Trueno Integration
// =============================================================================

#[test]
fn test_learning_rate_scheduler_vector() {
    let mut scheduler = LearningRateScheduler::cosine_annealing(0.001, 100, 0.0);

    // Collect learning rates as vector
    let lrs: Vec<f32> = (0..100)
        .map(|_| {
            let lr = scheduler.get_lr() as f32;
            scheduler.step();
            lr
        })
        .collect();

    let vector = Vector::from_slice(&lrs);
    assert_eq!(vector.len(), 100);
}

#[test]
fn test_training_gradients_vector() {
    let _config = TrainerConfig::default()
        .learning_rate(0.001)
        .epochs(10);

    // Simulate gradients vector
    let param_count = 1000;
    let gradients: Vec<f32> = (0..param_count)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();

    let vector = Vector::from_slice(&gradients);
    assert_eq!(vector.len(), param_count);
}

#[test]
fn test_batch_loss_history_matrix() {
    let num_epochs = 20;
    let batches_per_epoch = 100;

    // Simulate loss history as 2D matrix
    let losses: Vec<f32> = (0..num_epochs * batches_per_epoch)
        .map(|i| 1.0 / (1.0 + (i as f32) * 0.01))
        .collect();

    let matrix = Matrix::from_vec(num_epochs, batches_per_epoch, losses).unwrap();
    assert_eq!(matrix.rows(), num_epochs);
    assert_eq!(matrix.cols(), batches_per_epoch);
}

// =============================================================================
// Inference + Trueno Integration
// =============================================================================

#[test]
fn test_batch_processing_matrix() {
    let config = BatchConfig::default().max_batch_size(32);

    // Simulate batch input matrix
    let batch_size = config.max_batch_size;
    let seq_len = 128;
    let input: Vec<f32> = (0..batch_size * seq_len)
        .map(|i| (i % 1000) as f32)
        .collect();

    let matrix = Matrix::from_vec(batch_size, seq_len, input).unwrap();
    assert_eq!(matrix.rows(), batch_size);
    assert_eq!(matrix.cols(), seq_len);
}

#[test]
fn test_pipeline_output_matrix() {
    let _config = PipelineConfig::default();

    // Simulate pipeline output logits
    let batch_size = 8;
    let num_classes = 10;
    let logits: Vec<f32> = (0..batch_size * num_classes)
        .map(|i| ((i as f32) - 40.0) / 10.0)
        .collect();

    let matrix = Matrix::from_vec(batch_size, num_classes, logits).unwrap();
    assert_eq!(matrix.rows(), batch_size);
    assert_eq!(matrix.cols(), num_classes);
}

// =============================================================================
// Evaluation + Trueno Integration
// =============================================================================

#[test]
fn test_confusion_matrix() {
    let num_classes = 5;

    // Simulate confusion matrix
    let cm: Vec<f32> = (0..num_classes * num_classes)
        .map(|i| if i % (num_classes + 1) == 0 { 100.0 } else { 5.0 })
        .collect();

    let matrix = Matrix::from_vec(num_classes, num_classes, cm).unwrap();
    assert_eq!(matrix.rows(), num_classes);
    assert_eq!(matrix.cols(), num_classes);
}

#[test]
fn test_cross_validation_scores_vector() {
    let config = CrossValidationConfig::default();
    let n_folds = config.n_folds;

    // Simulate fold scores
    let scores: Vec<f32> = (0..n_folds)
        .map(|i| 0.85 + (i as f32) * 0.02)
        .collect();

    let vector = Vector::from_slice(&scores);
    assert_eq!(vector.len(), n_folds);
}

#[test]
fn test_metrics_per_class_vector() {
    let _metrics = ClassificationMetrics::default();
    let num_classes = 10;

    // Simulate per-class F1 scores
    let f1: Vec<f32> = (0..num_classes)
        .map(|i| 0.85 + (i as f32) * 0.01)
        .collect();

    let vector = Vector::from_slice(&f1);
    assert_eq!(vector.len(), num_classes);
}

// =============================================================================
// Deployment + Trueno Integration
// =============================================================================

#[test]
fn test_quantized_weights_vector() {
    let _config = QuantizationConfig::default()
        .quantization_type(QuantizationType::Int8);

    // Simulate quantized weights (stored as f32 but representing int8 values)
    let num_weights = 1000;
    let quantized: Vec<f32> = (0..num_weights)
        .map(|i| ((i % 256) as i16 - 128) as f32)
        .collect();

    let vector = Vector::from_slice(&quantized);
    assert_eq!(vector.len(), num_weights);
}

#[test]
fn test_scale_zero_point_vectors() {
    // Quantization parameters per channel
    let num_channels = 64;

    let scales: Vec<f32> = (0..num_channels)
        .map(|i| 0.01 + (i as f32) * 0.001)
        .collect();
    let zero_points: Vec<f32> = (0..num_channels)
        .map(|i| (i % 10) as f32)
        .collect();

    let scale_vector = Vector::from_slice(&scales);
    let zp_vector = Vector::from_slice(&zero_points);

    assert_eq!(scale_vector.len(), num_channels);
    assert_eq!(zp_vector.len(), num_channels);
}

// =============================================================================
// Matrix Operations Integration
// =============================================================================

#[test]
fn test_matrix_multiply_for_linear_layer() {
    // Simulate linear layer: y = xW
    let batch_size = 4;
    let in_features = 64;
    let out_features = 32;

    let x: Vec<f32> = (0..batch_size * in_features)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let w: Vec<f32> = (0..in_features * out_features)
        .map(|i| ((i as f32) - 1000.0) * 0.001)
        .collect();

    let x_matrix = Matrix::from_vec(batch_size, in_features, x).unwrap();
    let w_matrix = Matrix::from_vec(in_features, out_features, w).unwrap();

    assert_eq!(x_matrix.rows(), batch_size);
    assert_eq!(x_matrix.cols(), in_features);
    assert_eq!(w_matrix.rows(), in_features);
    assert_eq!(w_matrix.cols(), out_features);

    // Matrix multiplication would produce [batch_size, out_features]
    let result = x_matrix.matmul(&w_matrix).unwrap();
    assert_eq!(result.rows(), batch_size);
    assert_eq!(result.cols(), out_features);
}

#[test]
fn test_softmax_output_vector() {
    let num_classes = 100;

    // Simulate softmax outputs (probabilities sum to 1)
    let probs: Vec<f32> = (0..num_classes)
        .map(|i| if i == 0 { 0.9 } else { 0.1 / ((num_classes - 1) as f32) })
        .collect();

    let vector = Vector::from_slice(&probs);
    assert_eq!(vector.len(), num_classes);

    // Verify sum is approximately 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_layer_norm_stats_vectors() {
    let hidden_size = 256;

    // Simulate layer norm running stats
    let mean: Vec<f32> = (0..hidden_size).map(|_| 0.0).collect();
    let var: Vec<f32> = (0..hidden_size).map(|_| 1.0).collect();

    let mean_vector = Vector::from_slice(&mean);
    let var_vector = Vector::from_slice(&var);

    assert_eq!(mean_vector.len(), hidden_size);
    assert_eq!(var_vector.len(), hidden_size);
}

#[test]
fn test_vector_dot_product() {
    let dim = 128;

    let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..dim).map(|i| ((dim - i) as f32) * 0.1).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let dot = vec_a.dot(&vec_b).unwrap();
    assert!(dot > 0.0);
}

#[test]
fn test_matrix_transpose() {
    let rows = 4;
    let cols = 8;

    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let matrix = Matrix::from_vec(rows, cols, data).unwrap();
    let transposed = matrix.transpose();

    assert_eq!(transposed.rows(), cols);
    assert_eq!(transposed.cols(), rows);
}

// =============================================================================
// End-to-End Pipeline with Trueno
// =============================================================================

#[test]
fn test_full_pipeline_with_trueno() {
    // 1. Tokenize text
    let tokenizer = Tokenizer::new(TokenizerConfig::default());
    let tokens = tokenizer.tokenize("Integration test with trueno SIMD operations");

    // 2. Create feature matrix (bag of words style)
    let vocab_size = 1000;
    let num_samples = 1;
    let mut features = vec![0.0f32; num_samples * vocab_size];
    for token in &tokens {
        let idx = token.text.len() % vocab_size; // Simple hash
        features[idx] += 1.0;
    }

    let feature_matrix = Matrix::from_vec(num_samples, vocab_size, features).unwrap();
    assert_eq!(feature_matrix.rows(), num_samples);
    assert_eq!(feature_matrix.cols(), vocab_size);

    // 3. Simulate weight matrix
    let num_classes = 3;
    let weights: Vec<f32> = (0..vocab_size * num_classes)
        .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
        .collect();
    let weight_matrix = Matrix::from_vec(vocab_size, num_classes, weights).unwrap();

    // 4. Compute logits via matmul
    let logits = feature_matrix.matmul(&weight_matrix).unwrap();
    assert_eq!(logits.rows(), num_samples);
    assert_eq!(logits.cols(), num_classes);
}
