use serde_json;

mod common;

use common::*;
use tokenizers::models::wordlevel::WordLevel;

#[test]
fn bpe_serde() {
    let bpe = get_byte_level_bpe();
    let ser = serde_json::to_string(&bpe).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(bpe, de);
}

#[test]
fn wordpiece_serde() {
    let wordpiece = get_bert_wordpiece();
    let ser = serde_json::to_string(&wordpiece).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(wordpiece, de);
}

#[test]
fn wordlevel_serde() {
    let wordlevel = WordLevel::from_files("data/gpt2-vocab.json", "<unk>".into()).unwrap();
    let ser = serde_json::to_string(&wordlevel).unwrap();
    let de = serde_json::from_str(&ser).unwrap();
    assert_eq!(wordlevel, de);
}
