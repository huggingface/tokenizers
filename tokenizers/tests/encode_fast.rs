mod common;

use common::*;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::{AddedToken, SplitDelimiterBehavior, Tokenizer};
use tokenizers::TruncationParams;

static LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn inputs() -> Vec<&'static str> {
    vec![
        "Hello world!",
        "  leading and trailing spaces  ",
        "My name is John.\nI live in New-York, it's been 3 years already...",
        "no_spaces_just_one_long_token_1234567890",
        "unicode: caf\u{e9} na\u{ef}ve \u{1f600}\u{1f680} \u{4f60}\u{597d}\u{4e16}\u{754c} \u{30c8}\u{30fc}\u{30af}\u{30ca}\u{30a4}\u{30b6}\u{30fc}",
        "tabs\tand\r\nnewlines\n\n\nmixed   whitespace",
        "numbers 123 4567 89012345 and punct !!! ??? ...",
        "a",
        " ",
        "\n",
        "",
        "combining: e\u{301} A\u{30a} \u{1e69}",
    ]
}

fn assert_fast_matches_slow(tokenizer: &Tokenizer, input: &str) {
    let slow = tokenizer.encode(input, false).unwrap();
    let fast = tokenizer.encode_fast(input, false).unwrap();
    assert_eq!(
        fast.get_ids(),
        slow.get_ids(),
        "ids mismatch for input {input:?}"
    );
    assert!(fast.get_tokens().iter().all(|t| t.is_empty()));
    assert!(fast.get_offsets().iter().all(|o| *o == (0, 0)));
    assert!(fast.get_word_ids().iter().all(|w| w.is_none()));
    assert_eq!(fast.get_attention_mask(), vec![1; fast.len()]);
}

fn assert_all_inputs(tokenizer: &Tokenizer) {
    for input in inputs() {
        assert_fast_matches_slow(tokenizer, input);
    }
}

fn llama3_style_pre_tokenizer() -> Sequence {
    Sequence::new(vec![
        PreTokenizerWrapper::Split(
            Split::new(
                SplitPattern::Regex(LLAMA3_PATTERN.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ])
}

#[test]
fn gpt2_byte_level() {
    for add_prefix_space in [true, false] {
        let tokenizer = get_byte_level(add_prefix_space, false);
        assert_all_inputs(&tokenizer);
    }
}

#[test]
fn llama3_style_split_then_byte_level() {
    let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
    tokenizer.with_pre_tokenizer(Some(llama3_style_pre_tokenizer()));
    assert_all_inputs(&tokenizer);
}

#[test]
fn split_all_behaviors_and_invert() {
    use SplitDelimiterBehavior::*;
    for behavior in [
        Removed,
        Isolated,
        MergedWithPrevious,
        MergedWithNext,
        Contiguous,
    ] {
        for invert in [false, true] {
            let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
            tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
                PreTokenizerWrapper::Split(
                    Split::new(SplitPattern::Regex(r"\s+".to_owned()), behavior, invert).unwrap(),
                ),
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
            ])));
            assert_all_inputs(&tokenizer);
        }
    }
}

#[test]
fn with_added_tokens() {
    let mut tokenizer = get_byte_level(true, false);
    tokenizer
        .add_special_tokens([AddedToken::from("<|special|>", true)])
        .unwrap();
    tokenizer
        .add_tokens([AddedToken::from("John", false)])
        .unwrap();
    assert_fast_matches_slow(&tokenizer, "Hello <|special|> my name is John!");
    assert_fast_matches_slow(&tokenizer, "<|special|>");
    assert_fast_matches_slow(&tokenizer, "JohnJohn <|special|><|special|>");
    assert_all_inputs(&tokenizer);
}

#[test]
fn with_normalizer() {
    let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
    tokenizer.with_normalizer(Some(NFC)).unwrap();
    tokenizer.with_pre_tokenizer(Some(llama3_style_pre_tokenizer()));
    assert_all_inputs(&tokenizer);
}

#[test]
fn unsupported_pre_tokenizer_falls_back() {
    let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    assert_all_inputs(&tokenizer);

    let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Whitespace(Whitespace),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ])));
    assert_all_inputs(&tokenizer);
}

#[test]
fn no_pre_tokenizer() {
    let tokenizer = Tokenizer::new(get_byte_level_bpe());
    assert_fast_matches_slow(&tokenizer, "Hello");
    assert_fast_matches_slow(&tokenizer, "");
}

#[test]
fn with_truncation() {
    let mut tokenizer = get_byte_level(true, false);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 5,
            ..Default::default()
        }))
        .unwrap();
    assert_all_inputs(&tokenizer);
}

#[test]
fn pair_and_pre_tokenized_inputs() {
    let tokenizer = get_byte_level(true, false);
    let slow = tokenizer
        .encode(("Sequence A", "Sequence B"), false)
        .unwrap();
    let fast = tokenizer
        .encode_fast(("Sequence A", "Sequence B"), false)
        .unwrap();
    assert_eq!(fast.get_ids(), slow.get_ids());

    let slow = tokenizer
        .encode(&["Single", "sequence", "elements"][..], false)
        .unwrap();
    let fast = tokenizer
        .encode_fast(&["Single", "sequence", "elements"][..], false)
        .unwrap();
    assert_eq!(fast.get_ids(), slow.get_ids());
}

#[test]
fn unknown_bytes_bpe() {
    let tokenizer = Tokenizer::new(BPE::default());
    let fast = tokenizer.encode_fast("anything", false).unwrap();
    assert!(fast.get_ids().is_empty());
}

#[test]
fn llama3_real_tokenizer_on_big_text() {
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json")
        .expect("run `make test` to download data");
    let big = std::fs::read_to_string("data/small.txt").unwrap();
    for chunk in [&big[..], &big[..4096], "Hello there \u{1f600}"] {
        assert_fast_matches_slow(&tokenizer, chunk);
    }
}
