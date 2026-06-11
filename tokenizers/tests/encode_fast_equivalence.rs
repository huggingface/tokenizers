//! encode_fast must produce the same ids, type_ids and masks as encode,
//! across real tokenizer pipelines. encode_fast does not promise offsets,
//! token strings or word ids — those are not compared here.

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

/// Every other pre-tokenizer, swapped into the bert-wiki WordPiece pipeline.
#[test]
fn pre_tokenizer_sweep() {
    use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
    use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
    use tokenizers::pre_tokenizers::digits::Digits;
    use tokenizers::pre_tokenizers::fixed_length::FixedLength;
    use tokenizers::pre_tokenizers::punctuation::Punctuation;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::pre_tokenizers::unicode_scripts::UnicodeScripts;
    use tokenizers::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::SplitDelimiterBehavior::*;

    let pre_tokenizers: Vec<(&str, PreTokenizerWrapper)> = vec![
        ("bert", BertPreTokenizer.into()),
        ("whitespace", Whitespace.into()),
        ("whitespace_split", WhitespaceSplit.into()),
        ("punctuation_isolated", Punctuation::new(Isolated).into()),
        ("punctuation_removed", Punctuation::new(Removed).into()),
        (
            "punctuation_merged_prev",
            Punctuation::new(MergedWithPrevious).into(),
        ),
        (
            "punctuation_merged_next",
            Punctuation::new(MergedWithNext).into(),
        ),
        (
            "punctuation_contiguous",
            Punctuation::new(Contiguous).into(),
        ),
        ("digits_grouped", Digits::new(false).into()),
        ("digits_individual", Digits::new(true).into()),
        ("char_delimiter", CharDelimiterSplit::new(' ').into()),
        ("fixed_length", FixedLength::new(4).into()),
        ("unicode_scripts", UnicodeScripts::new().into()),
        (
            "split_regex",
            Split::new(SplitPattern::Regex(r"\w+|[^\w\s]+".into()), Isolated, false)
                .unwrap()
                .into(),
        ),
        (
            "split_invert",
            Split::new(SplitPattern::Regex(r"\s+".into()), Removed, true)
                .unwrap()
                .into(),
        ),
        (
            "sequence_whitespace_digits",
            Sequence::new(vec![Whitespace.into(), Digits::new(true).into()]).into(),
        ),
    ];

    let inputs = test_inputs();
    for (name, pre_tokenizer) in pre_tokenizers {
        eprintln!("pre_tokenizer: {name}");
        let mut tokenizer = Tokenizer::from_file("data/bert-wiki.json").unwrap();
        let _ = tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        check_tokenizer(&tokenizer, &inputs);
    }
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

/// Metaspace(First) prepends "▁" only to the first split. The vocab has both
/// "▁hello" and "hello", so a wrong prepend changes the ids.
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

/// A normalizer that removes leading characters (Strip) must not make
/// encode and encode_fast disagree. Metaspace(First) decides "first piece"
/// by split index, so both paths prepend "▁" here.
#[test]
fn metaspace_first_with_leading_strip() {
    use tokenizers::normalizers::Strip;

    let tokenizer = metaspace_first_tokenizer(Strip::new(true, true).into());

    let slow = tokenizer.encode("  hello world", false).unwrap();
    let fast = tokenizer.encode_fast("  hello world", false).unwrap();

    assert_eq!(slow.get_ids(), [0, 2], "▁hello, ▁world");
    assert_eq!(slow.get_ids(), fast.get_ids());
}
