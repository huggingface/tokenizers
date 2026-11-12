use serde_json::json;
use tk_encode::pipeline::{
    EncodeOptions, PipelinePreTokenizer, PreTokenizer, PreTokenizerScratch, Span,
};
use tk_encode::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::fixed_length::FixedLength;
use tk_encode::pre_tokenizers::sequence::PipelineSequence;
use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
use tk_encode::tokenizer::SplitDelimiterBehavior;
use tk_encode::utils::DEEPSEEK_PATTERNS;

fn sequence(tail: Vec<PipelinePreTokenizer>) -> PipelineSequence {
    let mut members: Vec<_> = DEEPSEEK_PATTERNS
        .into_iter()
        .map(|pattern| {
            PipelinePreTokenizer::Split(
                Split::native(
                    SplitPattern::Regex(pattern.into()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap(),
            )
        })
        .collect();
    members.extend(tail);
    PipelineSequence::new(members)
}

fn spans(sequence: &PipelineSequence, text: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    sequence
        .pre_tokenize(text, &mut PreTokenizerScratch::default(), &mut spans)
        .unwrap();
    spans
}

fn pieces(sequence: &PipelineSequence, text: &str) -> Vec<String> {
    spans(sequence, text)
        .iter()
        .map(|span| text[span.range()].into())
        .collect()
}

#[test]
fn trailing_digits_split_native_numeric_groups() {
    let seq = sequence(vec![PipelinePreTokenizer::Digits(Digits::new(true))]);
    assert_eq!(pieces(&seq, "hello123456"), ["hello", "1", "2", "3", "4", "5", "6"]);
}

#[test]
fn trailing_delimiters_are_removed() {
    let seq = sequence(vec![PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new('l'))]);
    assert_eq!(pieces(&seq, "hello123"), ["he", "o", "123"]);
}

#[test]
fn trailing_fixed_length_preserves_unicode_boundaries() {
    let seq = sequence(vec![PipelinePreTokenizer::FixedLength(FixedLength::new(1))]);
    let text = "éab東🙂1234";
    let actual = spans(&seq, text);
    let expected: Vec<_> = text
        .char_indices()
        .map(|(start, character)| Span {
            start: start as u32,
            end: (start + character.len_utf8()) as u32,
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn multiple_trailing_children_run_in_order() {
    let seq = sequence(vec![
        PipelinePreTokenizer::None,
        PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new('l')),
        PipelinePreTokenizer::FixedLength(FixedLength::new(1)),
    ]);
    assert_eq!(pieces(&seq, "hello123"), ["h", "e", "o", "1", "2", "3"]);
}

#[test]
fn native_prefix_with_only_identity_children_is_unchanged() {
    for tail in [vec![], vec![PipelinePreTokenizer::None, PipelinePreTokenizer::None]] {
        assert_eq!(pieces(&sequence(tail), "hello123456"), ["hello", "123", "456"]);
    }
}

#[test]
fn output_appends_and_scratch_can_be_reused() {
    let seq = sequence(vec![PipelinePreTokenizer::FixedLength(FixedLength::new(1))]);
    let mut scratch = PreTokenizerScratch::default();
    for text in ["éab123", "xy", "", "東🙂", "hello123456"] {
        let sentinel = Span { start: 0, end: 0 };
        let mut actual = vec![sentinel];
        seq.pre_tokenize(text, &mut scratch, &mut actual).unwrap();
        assert_eq!(actual[0], sentinel);
        let expected: Vec<_> = text.chars().map(|c| c.to_string()).collect();
        let actual: Vec<_> = actual[1..].iter().map(|span| text[span.range()].to_owned()).collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn errors_from_trailing_children_are_not_suppressed() {
    let unsupported = Split::native(
        SplitPattern::Regex(r"\d+".into()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .unwrap();
    let seq = sequence(vec![PipelinePreTokenizer::Split(unsupported)]);
    let error = seq
        .pre_tokenize("123", &mut PreTokenizerScratch::default(), &mut Vec::new())
        .unwrap_err();
    assert!(error.to_string().contains("backend"));
}

#[test]
fn canonical_json_pipeline_applies_the_configured_tail() {
    let mut members: Vec<_> = DEEPSEEK_PATTERNS
        .iter()
        .map(|pattern| json!({
            "type": "Split", "pattern": {"Regex": pattern},
            "behavior": "Isolated", "invert": false
        }))
        .collect();
    members.push(json!({"type": "Digits", "individual_digits": true}));
    let config = json!({
        "version": "2.0", "added_tokens": [],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": members},
        "model": {"type": "WordLevel", "unk_token": "<unk>",
                  "vocab": {"<unk>": 0, "abc": 1, "1": 2, "2": 3, "3": 4, "123": 5}}
    }).to_string();
    let tokenizer = tk_serialize::from_json(&config).unwrap();
    let encoded = tokenizer.encode("abc123", &EncodeOptions::no_specials()).wait().unwrap();
    let ids: Vec<_> = encoded[0].ids().iter().map(|token| token.id()).collect();
    assert_eq!(ids, [1, 2, 3, 4]);
}

#[test]
fn ordinary_sequences_keep_their_existing_behavior() {
    let seq = PipelineSequence::new(vec![
        PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new('|')),
        PipelinePreTokenizer::Digits(Digits::new(true)),
    ]);
    assert_eq!(pieces(&seq, "ab12|cd3"), ["ab", "1", "2", "cd", "3"]);
}

