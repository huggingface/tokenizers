//! encode_fast must produce the same ids/type_ids/special-tokens-mask as encode,
//! across real tokenizer pipelines. Offsets, token strings and word ids are
//! explicitly out of contract for encode_fast.

use tokenizers::tokenizer::EncodeInput;
use tokenizers::{AddedToken, Tokenizer};

fn test_inputs() -> Vec<String> {
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let mut inputs: Vec<String> = data.lines().take(200).map(|s| s.to_string()).collect();
    inputs.extend(
        [
            "",
            " ",
            "    ",
            "Hello world",
            " leading space",
            "trailing space ",
            "punct!!! ... and 123 numbers",
            "unicode: héllo wörld précis ça",
            "CJK: 你好世界 こんにちは 안녕하세요",
            "emoji: 👋🌍 family: 👨‍👩‍👧‍👦",
            "combining: e\u{301} a\u{30A} ﬀ ligature",
            "mixed العربية and עברית rtl",
            "tabs\tand\nnewlines\r\nhere",
            "'s 't 're 've 'm 'll 'd contractions",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    inputs
}

fn check_tokenizer(tokenizer: &Tokenizer, inputs: &[String]) {
    for input in inputs {
        let slow = tokenizer.encode(input.as_str(), true).unwrap();
        let fast = tokenizer.encode_fast(input.as_str(), true).unwrap();
        assert_eq!(slow.get_ids(), fast.get_ids(), "ids differ on {input:?}");
        assert_eq!(
            slow.get_type_ids(),
            fast.get_type_ids(),
            "type_ids differ on {input:?}"
        );
        assert_eq!(
            slow.get_special_tokens_mask(),
            fast.get_special_tokens_mask(),
            "special tokens mask differs on {input:?}"
        );
        assert_eq!(
            slow.get_attention_mask(),
            fast.get_attention_mask(),
            "attention mask differs on {input:?}"
        );
    }

    // Pair encoding
    let slow = tokenizer
        .encode(("First sequence", "Second sequence"), true)
        .unwrap();
    let fast = tokenizer
        .encode_fast(("First sequence", "Second sequence"), true)
        .unwrap();
    assert_eq!(slow.get_ids(), fast.get_ids());
    assert_eq!(slow.get_type_ids(), fast.get_type_ids());

    // Batch
    let batch: Vec<EncodeInput> = inputs.iter().map(|s| s.as_str().into()).collect();
    let slow_batch = tokenizer.encode_batch(batch.clone(), true).unwrap();
    let fast_batch = tokenizer.encode_batch_fast(batch, true).unwrap();
    assert_eq!(slow_batch.len(), fast_batch.len());
    for (i, (slow, fast)) in slow_batch.iter().zip(fast_batch.iter()).enumerate() {
        assert_eq!(slow.get_ids(), fast.get_ids(), "batch ids differ at {i}");
    }
}

#[test]
fn gpt2_with_added_tokens() {
    let mut tokenizer = Tokenizer::from_file("data/roberta.json").unwrap();
    tokenizer
        .add_tokens(vec![AddedToken::from("procedural", true)])
        .unwrap();
    tokenizer
        .add_special_tokens(vec![AddedToken::from("<custom>", true)
            .lstrip(true)
            .rstrip(true)])
        .unwrap();

    let mut inputs = test_inputs();
    inputs.push("a procedural text with <custom> tokens procedural".into());
    inputs.push("  <custom>  spaces around <custom>".into());
    check_tokenizer(&tokenizer, &inputs);
}

#[test]
fn llama3() {
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    check_tokenizer(&tokenizer, &test_inputs());
}

#[test]
fn bert_wiki() {
    let tokenizer = Tokenizer::from_file("data/bert-wiki.json").unwrap();
    check_tokenizer(&tokenizer, &test_inputs());
}

#[test]
fn albert_metaspace() {
    let tokenizer = Tokenizer::from_file("data/albert-base-v1-tokenizer.json").unwrap();
    check_tokenizer(&tokenizer, &test_inputs());
}
