//! The reader's own tests: one case per component it reads, plus the refusals that hand a file to
//! tk-convert instead.

use super::decoders::read_one_decoder;
use super::*;
use tk_encode::tokenizer::{PaddingDirection, PaddingStrategy};

/// A minimal BPE that needs no data files: two merges over a four-token vocab.
const TINY_BPE: &str = r#"{"type": "BPE", "byte_level": false, "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
    "merges": [["a", "b"], ["ab", "ab"]]}"#;

/// A whole `tokenizer.json`. Every component is `null` and the model is [`TINY_BPE`] unless
/// `overrides` names it, so a test states the document it reads rather than patching one by text.
fn config<'a>(overrides: &'a [(&'a str, &'a str)]) -> String {
    let field = |name: &str, default: &'a str| -> &'a str {
        overrides
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(default, |(_, json)| *json)
    };
    format!(
        r#"{{"version": "2.0", "added_tokens": {}, "normalizer": {}, "pre_tokenizer": {},
            "post_processor": {}, "decoder": {}, "model": {}, "padding": {}}}"#,
        field("added_tokens", "[]"),
        field("normalizer", "null"),
        field("pre_tokenizer", "null"),
        field("post_processor", "null"),
        field("decoder", "null"),
        field("model", TINY_BPE),
        field("padding", "null"),
    )
}

/// The message from a config the reader must refuse. Not `unwrap_err`: `PipelineTokenizer` has no
/// `Debug`, which is what that would need on the `Ok` side.
fn read_err(text: &str) -> String {
    match from_json(text) {
        Ok(_) => panic!("expected the reader to refuse this config"),
        Err(e) => e.to_string(),
    }
}

fn ids(text: &str, input: &str) -> Vec<u32> {
    from_json(text)
        .expect("the config reads")
        .encode(input, true)
        .wait()
        .unwrap()
        .iter()
        .flat_map(|e| e.ids())
        .map(|t| t.id())
        .collect()
}

/// `(slot, json, text, ids)`: what a component does to an encode, which is the only thing a caller
/// can observe. `ab` scoring better than `a`+`a` is what makes the Unigram row a lattice test.
#[rustfmt::skip]
const ENCODES: &[(&str, &str, &str, &[u32])] = &[
    ("model", TINY_BPE, "abab", &[3]),
    // [CLS] $A [SEP] around the single sequence, from either spelling of the frame.
    ("post_processor", r#"{"type": "BertProcessing", "sep": ["b", 1], "cls": ["a", 0]}"#, "abab", &[0, 3, 1]),
    ("post_processor", r#"{"type": "RobertaProcessing", "sep": ["b", 1], "cls": ["a", 0],
        "trim_offsets": true, "add_prefix_space": true}"#, "abab", &[0, 3, 1]),
];

#[cfg(feature = "unigram")]
#[rustfmt::skip]
const UNIGRAM: &[(&str, &str, &str, &[u32])] = &[
    ("model", r#"{"type": "Unigram", "unk_id": 0, "byte_fallback": false,
        "vocab": [["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]]}"#, "ab", &[2]),
];
#[cfg(not(feature = "unigram"))]
const UNIGRAM: &[(&str, &str, &str, &[u32])] = &[];

#[cfg(feature = "wordpiece")]
#[rustfmt::skip]
const WORDPIECE: &[(&str, &str, &str, &[u32])] = &[
    ("model", r#"{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "@@",
        "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "ab": 1, "@@c": 2}}"#, "abc", &[1, 2]),
];
#[cfg(not(feature = "wordpiece"))]
const WORDPIECE: &[(&str, &str, &str, &[u32])] = &[];

#[cfg(feature = "wordlevel")]
#[rustfmt::skip]
const WORDLEVEL: &[(&str, &str, &str, &[u32])] = &[
    ("model", r#"{"type": "WordLevel", "unk_token": "<unk>", "vocab": {"<unk>": 0, "hello": 1}}"#, "hello", &[1]),
];
#[cfg(not(feature = "wordlevel"))]
const WORDLEVEL: &[(&str, &str, &str, &[u32])] = &[];

#[test]
fn every_component_encodes_what_it_should() {
    for (slot, json, text, expected) in ENCODES
        .iter()
        .chain(UNIGRAM)
        .chain(WORDPIECE)
        .chain(WORDLEVEL)
    {
        assert_eq!(
            ids(&config(&[(slot, json)]), text),
            *expected,
            "{slot}: {json}"
        );
    }
}

/// `(slot, json, in the message)`: a legacy or malformed shape, and what the refusal has to name so
/// the caller knows what to convert. Inferring a model kind from its keys, rewriting `"a b"` merges,
/// and folding a `Metaspace` or `ByteLevel` pre-tokenizer are all tk-convert's job now.
#[rustfmt::skip]
const REFUSED: &[(&str, &str, &str)] = &[
    ("model", r#"{"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]}"#, "`model` with no `type`"),
    ("model", r#"{"type": "BPE", "byte_level": false, "vocab": {"a": 0, "b": 1, "ab": 2},
        "merges": ["a b"]}"#, "[left, right] pair"),
    ("normalizer", r#"{"type": "Strip", "strip_left": true}"#, "strip_right"),
    ("normalizer", r#"{"type": "Invented"}"#, "`Invented` normalizer"),
    ("pre_tokenizer", r#"{"type": "ByteLevel", "use_regex": true}"#, "`ByteLevel` pre-tokenizer"),
    ("pre_tokenizer", r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"}"#,
        "`Metaspace` pre-tokenizer"),
    ("post_processor", r#"{"type": "BertProcessing", "sep": ["b"], "cls": ["a", 0]}"#, "[token, id] pair"),
    ("padding", r#"{"direction": "Right", "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"}"#, "no `strategy`"),
    ("padding", r#"{"strategy": "Invented", "direction": "Right", "pad_id": 0, "pad_type_id": 0,
        "pad_token": "[PAD]"}"#, "unknown padding strategy"),
    ("padding", r#"{"strategy": "BatchLongest", "direction": "Up", "pad_id": 0, "pad_type_id": 0,
        "pad_token": "[PAD]"}"#, "unknown padding direction"),
];

#[cfg(feature = "unigram")]
#[rustfmt::skip]
const REFUSED_UNIGRAM: &[(&str, &str, &str)] = &[
    ("model", r#"{"type": "Unigram", "vocab": [["a"]]}"#, "[token, score] pair"),
];
#[cfg(not(feature = "unigram"))]
const REFUSED_UNIGRAM: &[(&str, &str, &str)] = &[];

/// `AddedToken` and `WordPiece` have no defaults on either path, so a missing field is an error.
#[cfg(feature = "wordpiece")]
#[rustfmt::skip]
const REFUSED_WORDPIECE: &[(&str, &str, &str)] = &[
    ("model", r#"{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "@@",
        "vocab": {"[UNK]": 0, "ab": 1}}"#, "max_input_chars_per_word"),
];
#[cfg(not(feature = "wordpiece"))]
const REFUSED_WORDPIECE: &[(&str, &str, &str)] = &[];

/// `strip_accents: null` means "decide from `lowercase`", so the key has to be there.
#[cfg(feature = "normalizers")]
#[rustfmt::skip]
const REFUSED_BERT: &[(&str, &str, &str)] = &[
    ("normalizer", r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
        "lowercase": true}"#, "strip_accents"),
];
#[cfg(not(feature = "normalizers"))]
const REFUSED_BERT: &[(&str, &str, &str)] = &[];

#[test]
fn a_legacy_or_malformed_shape_is_refused_by_name() {
    for (slot, json, message) in REFUSED
        .iter()
        .chain(REFUSED_UNIGRAM)
        .chain(REFUSED_WORDPIECE)
        .chain(REFUSED_BERT)
    {
        let error = read_err(&config(&[(slot, json)]));
        assert!(error.contains(message), "{slot}: {json}\ngot: {error}");
    }
}

#[test]
fn padding_null_reads_as_no_padding() {
    let tokenizer = from_json(&config(&[])).unwrap();

    assert!(tokenizer.get_padding().is_none());
}

#[test]
fn padding_batch_longest_reads_back() {
    let padding = r#"{"strategy": "BatchLongest", "direction": "Right", "pad_to_multiple_of": null,
        "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"}"#;

    let tokenizer = from_json(&config(&[("padding", padding)])).unwrap();
    let params = tokenizer.get_padding().unwrap();

    assert!(matches!(params.strategy, PaddingStrategy::BatchLongest));
    assert!(matches!(params.direction, PaddingDirection::Right));
    assert_eq!(params.pad_to_multiple_of, None);
    assert_eq!(params.pad_id, 0);
    assert_eq!(params.pad_type_id, 0);
    assert_eq!(params.pad_token, "[PAD]");
}

#[test]
fn padding_fixed_reads_back() {
    let padding = r#"{"strategy": {"Fixed": 128}, "direction": "Left", "pad_to_multiple_of": 8,
        "pad_id": 3, "pad_type_id": 1, "pad_token": "<pad>"}"#;

    let tokenizer = from_json(&config(&[("padding", padding)])).unwrap();
    let params = tokenizer.get_padding().unwrap();

    assert!(matches!(params.strategy, PaddingStrategy::Fixed(128)));
    assert!(matches!(params.direction, PaddingDirection::Left));
    assert_eq!(params.pad_to_multiple_of, Some(8));
    assert_eq!(params.pad_id, 3);
    assert_eq!(params.pad_type_id, 1);
    assert_eq!(params.pad_token, "<pad>");
}

/// `crate::BASE64` is a spelled-out engine rather than `STANDARD`, purely so that decoding stays
/// indifferent to padding the way `spm_precompiled`'s `base64` 0.13 was. That is the only decision
/// left here -- swap the engine for `STANDARD` and this fails.
#[test]
fn a_charsmap_decodes_padded_or_not() {
    use base64::Engine as _;

    assert_eq!(crate::BASE64.decode("Zm9v").unwrap(), b"foo");
    assert_eq!(crate::BASE64.decode("Zg==").unwrap(), b"f");
    assert_eq!(crate::BASE64.decode("Zg").unwrap(), b"f", "unpadded");
    assert!(crate::BASE64.decode("Zm9v!").is_err(), "illegal character");
}

#[test]
fn flattens_a_normalizer_sequence_and_drops_an_empty_one() {
    let seq = config(&[(
        "normalizer",
        r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"},
            {"type": "Strip", "strip_left": true, "strip_right": true}]}"#,
    )]);
    let doc = Json::parse(&seq).unwrap();
    assert_eq!(read_normalizers(doc.field("normalizer")).unwrap().len(), 2);

    let empty = config(&[("normalizer", r#"{"type": "Sequence", "normalizers": []}"#)]);
    let doc = Json::parse(&empty).unwrap();
    assert!(
        read_normalizers(doc.field("normalizer"))
            .unwrap()
            .is_empty()
    );
}

/// The byte-level map is a model field, and the pre-tokenizer half it used to imply is gone: with
/// no `pre_tokenizer` there is no splitter at all, whatever the model says.
#[test]
fn byte_level_is_a_model_field_not_a_pre_tokenizer() {
    let json = config(&[(
        "model",
        r#"{"type": "BPE", "byte_level": true, "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
            "merges": [["a", "b"], ["ab", "ab"]]}"#,
    )]);
    let doc = Json::parse(&json).unwrap();
    let pretok = read_pre_tokenizer(doc.field("pre_tokenizer")).unwrap();
    assert!(matches!(pretok, PipelinePreTokenizer::None));
    // Where the flag *does* land is the model, which then wants every byte to be an atom. The
    // four-token vocab above is not one, so this is refused for that reason and no other.
    assert!(
        read_err(&json).contains("Byte atom"),
        "not the byte-atom refusal"
    );
}

/// One config per decoder variant, so a field rename fails here rather than silently producing a
/// decoder that decodes nothing.
#[test]
#[rustfmt::skip]
fn builds_every_decoder_variant_and_refuses_the_rest() {
    for json in [
        r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}"#,
        r#"{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}"#,
        r#"{"type": "ByteFallback"}"#,
        r#"{"type": "Fuse"}"#,
        r#"{"type": "Strip", "content": " ", "start": 1, "stop": 0}"#,
        r#"{"type": "BPEDecoder", "suffix": "</w>"}"#,
        r#"{"type": "WordPiece", "prefix": "@@", "cleanup": true}"#,
        r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"}"#,
        r#"{"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}"#,
        r#"{"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}"#,
    ] {
        let doc = Json::parse(json).unwrap();
        read_one_decoder(&doc).unwrap_or_else(|e| panic!("{json}: {e}"));
    }
    let doc = Json::parse(r#"{"type": "Invented"}"#).unwrap();
    let error = read_one_decoder(&doc).unwrap_err().to_string();
    assert!(error.contains("`Invented` decoder"), "{error}");
}

#[test]
fn a_decoder_reads_back_what_the_slim_path_wired_up() {
    let json = config(&[(
        "decoder",
        r#"{"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "Fuse"}]}"#,
    )]);
    let tok = from_json(&json).expect("the config reads");
    // `abab` is id 3; without a decoder the ids would join with a space instead.
    assert_eq!(tok.decode(&[3, 0], false).unwrap(), "ababa");
}

/// A Unigram score is read as `serde_json`'s `f64`, not the correctly-rounded one: `t5-base` spells
/// `▁` as `-2.0122928619384766`, correctly rounded `c000192d00000000`, and the vendored serde path
/// lands one ULP away on `c000192d00000001` -- as it does for 8334 of t5's 32100 scores. Scores
/// feed a Viterbi lattice, so reproducing serde bit for bit is what keeps the ids that ship today.
/// One ULP is also the *bound*: a real bug in the float path would miss by more, so this fails
/// either way -- if the deviation disappears, or if it grows.
#[test]
#[cfg(feature = "unigram")]
fn a_unigram_score_is_serdes_f64_one_ulp_off_the_correctly_rounded_one() {
    let json = config(&[(
        "model",
        r#"{"type": "Unigram", "unk_id": 0,
            "vocab": [["<unk>", 0.0], ["▁", -2.0122928619384766]]}"#,
    )]);
    let doc = Json::parse(&json).unwrap();
    let unigram = read_unigram(doc.field("model").unwrap()).unwrap();
    let (_, score) = unigram.iter().nth(1).expect("the second entry is `▁`");
    assert_eq!(score.to_bits(), 0xc000_192d_0000_0001, "serde's f64");
    let correctly_rounded = f64::from(*score as f32);
    assert_eq!(
        (score.to_bits() as i64).abs_diff(correctly_rounded.to_bits() as i64),
        1,
        "the score must sit exactly one ULP from the f32 the file encodes"
    );
}
