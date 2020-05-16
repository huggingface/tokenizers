use serde_json;

mod common;

use common::*;

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
