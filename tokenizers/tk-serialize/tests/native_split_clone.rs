use serde_json::json;
use std::collections::BTreeMap;
use tk_encode::pipeline::{
    EncodeOptions, PipelineTokenizer, PreTokenizer, PreTokenizerScratch,
};
use tk_encode::pre_tokenizers::split::{Search, Split, SplitPattern};
use tk_encode::tokenizer::SplitDelimiterBehavior;
use tk_encode::utils::DEEPSEEK_PATTERNS;
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary;

fn pieces(split: &Split, text: &str) -> Vec<String> {
    let mut spans = Vec::new();
    split
        .pre_tokenize(text, &mut PreTokenizerScratch::default(), &mut spans)
        .unwrap();
    spans.iter().map(|span| text[span.range()].into()).collect()
}

fn encode(tokenizer: &PipelineTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, &EncodeOptions::no_specials())
        .wait()
        .unwrap()[0]
        .ids()
        .iter()
        .map(|token| token.id())
        .collect()
}

#[test]
fn cloning_native_deepseek_members_preserves_their_backend() {
    for pattern in DEEPSEEK_PATTERNS {
        let original = Split::native(
            SplitPattern::Regex(pattern.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap();
        let cloned = original.clone();
        assert_eq!(cloned, original);
        assert!(matches!(cloned.search, Search::Unavailable));
        assert_eq!(cloned.grammar(), original.grammar());
    }
}

#[test]
fn cloned_canonical_deepseek_sequence_remains_usable_in_a_pipeline() {
    let pretokenizers: Vec<_> = DEEPSEEK_PATTERNS
        .iter()
        .map(|pattern| {
            json!({
                "type": "Split", "pattern": {"Regex": pattern},
                "behavior": "Isolated", "invert": false
            })
        })
        .collect();
    let config = json!({
        "version": "2.0", "added_tokens": [],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": pretokenizers},
        "model": {
            "type": "BPE", "byte_level": false, "merges": [],
            "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "1": 4, "2": 5,
                      "3": 6, "4": 7, "5": 8, "6": 9, "\n": 10}
        }
    })
    .to_string();
    let original = tk_serialize::from_json(&config).unwrap();
    let expected = [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(encode(&original, "hello123456\n"), expected);
    let cloned = PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        original.get_pre_tokenizer().clone(),
        original.get_model().clone(),
        Default::default(),
        None,
        BTreeMap::new(),
        None,
        None,
    );
    assert_eq!(encode(&cloned, "hello123456\n"), expected);
    assert_eq!(encode(&cloned, "123456hello"), [4, 5, 6, 7, 8, 9, 0, 1, 2, 2, 3]);
}

#[test]
fn cloning_a_native_unrecognized_regex_keeps_the_same_runtime_error() {
    let original = Split::native(
        SplitPattern::Regex(r"\d+".into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .unwrap();
    let cloned = original.clone();
    for split in [&original, &cloned] {
        let error = split
            .pre_tokenize("ab12", &mut PreTokenizerScratch::default(), &mut Vec::new())
            .unwrap_err();
        assert!(error.to_string().contains("backend"));
    }
}

#[test]
fn cloning_a_recognized_native_regex_preserves_splitting() {
    let original = Split::native(
        SplitPattern::Regex(bitcannon::regexes::GPT2.into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .unwrap();
    let cloned = original.clone();
    assert_eq!(pieces(&original, "Hello world!"), ["Hello", " world", "!"]);
    assert_eq!(pieces(&cloned, "Hello world!"), ["Hello", " world", "!"]);
}

#[test]
fn cloning_a_literal_split_preserves_unicode_and_flags() {
    for invert in [false, true] {
        let original = Split::new("|", SplitDelimiterBehavior::Removed, invert).unwrap();
        let cloned = original.clone();
        let expected = if invert { vec!["|"] } else { vec!["a", "é"] };
        assert_eq!(pieces(&original, "a|é"), expected);
        assert_eq!(pieces(&cloned, "a|é"), expected);
    }
}

#[cfg(feature = "fancy-regex")]
#[test]
fn cloning_a_backend_regex_preserves_matching() {
    let original = Split::new(
        SplitPattern::Regex(r"\d+".into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .unwrap();
    let cloned = original.clone();
    assert!(matches!(cloned.search, Search::Regex(_)));
    assert_eq!(pieces(&cloned, "ab12 cd"), ["ab", "12", " cd"]);
}
