#![cfg(feature = "byte_level_fast")]
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

fn load(f: &str) -> Tokenizer {
    Tokenizer::from_file(format!("data/{f}")).unwrap()
}

#[test]
fn enabled_for_byte_level_models() {
    // top-level ByteLevel: gpt2, roberta | nested in a Sequence: deepseek, llama3, glm, gpt-oss
    for config_file in [
        "gpt2-slim.json",
        "roberta-slim.json",
        "deepseek-v4-slim.json",
        "llama-3-slim.json",
        "glm-5.2-slim.json",
        "gpt-oss-slim.json",
    ] {
        assert!(
            load(config_file).byte_level_fast_enabled(),
            "{} must enable the fast path",
            config_file
        );
    }
}

#[test]
fn disabled_for_non_byte_level_models() {
    for config_file in ["gemma-4-slim.json", "bert-wiki-slim.json"] {
        assert!(
            !load(config_file).byte_level_fast_enabled(),
            "{} must NOT enable the fast path",
            config_file
        );
    }
}

#[test]
fn empty_sequence_normalizer_counts_as_noop() {
    // deepseek's normalizer is Sequence[] — must not disqualify
    assert!(load("deepseek-v4-slim.json").byte_level_fast_enabled());
}

#[test]
fn disabled_when_pretokenizer_swapped_out() {
    let mut tok = load("gpt2-slim.json");
    assert!(tok.byte_level_fast_enabled());
    tok.with_pre_tokenizer(Some(Whitespace::default())); // auto-refresh
    assert!(!tok.byte_level_fast_enabled());
}

#[test]
fn disabled_when_real_normalizer_added() {
    let mut tok = load("deepseek-v4-slim.json");
    assert!(tok.byte_level_fast_enabled());
    tok.with_normalizer(Some(NFC)).unwrap(); // auto-refresh
    assert!(!tok.byte_level_fast_enabled());
}
