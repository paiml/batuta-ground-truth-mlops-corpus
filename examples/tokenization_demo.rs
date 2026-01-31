//! Tokenization example demonstrating preprocessing capabilities

use batuta_ground_truth_mlops_corpus::preprocessing::{
    Tokenizer, TokenizerConfig, TokenizerType,
};

fn main() {
    println!("=== Tokenization Demo ===\n");

    // Whitespace tokenizer (default)
    let tokenizer = Tokenizer::default();
    let text = "Hello, world! This is a test.";
    let tokens = tokenizer.tokenize(text);
    println!("Whitespace tokenization of: \"{text}\"");
    for token in &tokens {
        println!("  - '{}' [{}, {}]", token.text, token.start, token.end);
    }

    // BERT-style WordPiece tokenizer
    println!("\nWordPiece tokenization:");
    let bert_tokenizer = Tokenizer::new(TokenizerConfig::bert());
    let tokens = bert_tokenizer.tokenize("Understanding transformers");
    for token in &tokens {
        println!("  - '{}' [{}, {}]", token.text, token.start, token.end);
    }

    // Custom configuration
    println!("\nCustom configuration:");
    let custom_config = TokenizerConfig::default()
        .max_length(5)
        .lowercase(false)
        .tokenizer_type(TokenizerType::Whitespace);
    let custom_tokenizer = Tokenizer::new(custom_config);
    let tokens = custom_tokenizer.tokenize("ONE TWO THREE FOUR FIVE SIX SEVEN");
    println!("Truncated to 5 tokens:");
    for token in &tokens {
        println!("  - '{}'", token.text);
    }
}
