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

fn metaspace_first_tokenizer(normalizer: tokenizers::NormalizerWrapper) -> Tokenizer {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

    let vocab: ahash::AHashMap<String, u32> = [
        "▁hello", "hello", "▁world", "world", "▁how", "how", "▁are", "are", "▁you", "you",
        "<mask>", "<unk>", "▁",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| (s.to_string(), i as u32))
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".into())
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(model);
    let _ = tokenizer.with_normalizer(Some(normalizer));
    let _ = tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::First, true)));
    tokenizer
}

/// Metaspace(First) prepends only to the piece at original position 0 — the
/// one consumer of original-referential offsets on the fast path. The vocab
/// distinguishes prepended from bare tokens so any divergence shows in ids.
///
/// Known limitation, pre-dating the unaligned mode: a normalizer that strips
/// leading content from the sequence start makes encode_fast prepend where
/// encode does not — pinned by `metaspace_first_leading_strip_known_divergence`
/// below. No such normalizer is used here.
#[test]
fn metaspace_first_prepend_scheme() {
    use tokenizers::normalizers::utils::Lowercase;

    let mut tokenizer = metaspace_first_tokenizer(Lowercase.into());
    tokenizer
        .add_special_tokens(vec![AddedToken::from("<mask>", true)])
        .unwrap();

    for input in [
        "Hello world how are you",
        "<mask> hello world",
        "hello <mask> world",
        " hello world",
        "hello",
        "<mask>",
    ] {
        let slow = tokenizer.encode(input, false).unwrap();
        let fast = tokenizer.encode_fast(input, false).unwrap();
        assert_eq!(slow.get_ids(), fast.get_ids(), "ids differ on {input:?}");
        // The prepended form must actually occur, or this test proves nothing
        if input == "Hello world how are you" {
            assert_eq!(slow.get_ids()[0], 0, "expected ▁hello first");
            assert!(!slow.get_ids()[1..].contains(&0));
        }
    }
}

/// Canary for the KNOWN encode/encode_fast divergence: without alignments the
/// fast path cannot know a normalizer stripped leading content, so the first
/// piece keeps original_shift 0 and Metaspace(First) prepends where encode
/// does not. Introduced with `normalize_str` (the trivial-alignments
/// `set_normalized`), kept by the unaligned mode.
///
/// If this test starts failing because both sides agree, the limitation got
/// fixed: delete this test and the caveats referencing it (slice() comment in
/// normalizer.rs, doc of metaspace_first_prepend_scheme above).
#[test]
fn metaspace_first_leading_strip_known_divergence() {
    use tokenizers::normalizers::Strip;

    let tokenizer = metaspace_first_tokenizer(Strip::new(true, true).into());

    let slow = tokenizer.encode("  hello world", false).unwrap();
    let fast = tokenizer.encode_fast("  hello world", false).unwrap();

    // encode: stripped first piece maps to original position 2 → no prepend
    assert_eq!(slow.get_ids(), [1, 2], "hello, ▁world");
    // encode_fast: position information lost → prepend
    assert_eq!(fast.get_ids(), [0, 2], "▁hello, ▁world");
}
