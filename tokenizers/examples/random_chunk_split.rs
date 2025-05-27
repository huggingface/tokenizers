use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::random_chunk::RandomChunkSplit;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

fn main() {
    // The example text that contains multi-word expressions
    let text = "We want to learn multi-word expressions like 'New York' or 'machine learning'";
    println!("Original text: {}", text);

    // Demonstrate how pre-tokenization works
    println!("\n=== Pre-tokenization Demonstration ===");
    
    // WhitespaceSplit pre-tokenization
    let mut pretokenized_whitespace = PreTokenizedString::from(text);
    let whitespace_split = WhitespaceSplit;
    whitespace_split.pre_tokenize(&mut pretokenized_whitespace).unwrap();
    let splits_whitespace = pretokenized_whitespace
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (s.to_owned(), o))
        .collect::<Vec<_>>();
    println!("WhitespaceSplit splits:");
    for (token, offsets) in &splits_whitespace {
        println!("  '{}' at position {:?}", token, offsets);
    }
    println!("Number of splits: {}", splits_whitespace.len());
    
    // RandomChunkSplit pre-tokenization
    let mut pretokenized_random = PreTokenizedString::from(text);
    let random_chunk_split = RandomChunkSplit::new(2, 5);
    random_chunk_split.pre_tokenize(&mut pretokenized_random).unwrap();
    let splits_random = pretokenized_random
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (s.to_owned(), o))
        .collect::<Vec<_>>();
    println!("\nRandomChunkSplit splits (min_length=2, max_length=5):");
    for (token, offsets) in &splits_random {
        println!("  '{}' at position {:?}", token, offsets);
    }
    println!("Number of splits: {}", splits_random.len());
    
    // Show a more extreme example with large chunks
    let mut pretokenized_large = PreTokenizedString::from(text);
    let large_chunk_split = RandomChunkSplit::new(10, 15);
    large_chunk_split.pre_tokenize(&mut pretokenized_large).unwrap();
    let splits_large = pretokenized_large
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (s.to_owned(), o))
        .collect::<Vec<_>>();
    println!("\nRandomChunkSplit splits (min_length=10, max_length=15):");
    for (token, offsets) in &splits_large {
        println!("  '{}' at position {:?}", token, offsets);
    }
    println!("Number of splits: {}", splits_large.len());
}