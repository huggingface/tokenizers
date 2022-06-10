mod utils;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::Tokenizer;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn tokenize(string: &str) -> Vec<u32> {
    let vocab: Vocab = vec![
        ("a".to_string(), 0),
        ("##b".to_string(), 1),
        ("##c".to_string(), 2),
        ("ab".to_string(), 3),
        ("abc".to_string(), 4),
    ]
    .into_iter()
    .collect();

    let merges = vec![
        ("a".to_string(), "##b".to_string()),
        ("ab".to_string(), "##c".to_string()),
    ];

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token("[UNK]".to_string())
        .continuing_subword_prefix("##".to_string())
        .build()
        .unwrap();
    let tokenizer = Tokenizer::new(bpe);
    tokenizer
        .encode(string, false)
        .unwrap()
        .get_ids()
        .into_iter()
        .cloned()
        .collect()
}
