//! The reader's own tests: `TINY_BPE` plus one case per component it reads.

use super::decoders::read_one_decoder;
use super::*;

/// A minimal BPE that needs no data files: two merges over a four-token vocab.
const TINY_BPE: &str = r#"{"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
    "merges": [["a", "b"], ["ab", "ab"]]}"#;

/// A whole `tokenizer.json`. Every component is `null` and the model is [`TINY_BPE`] unless
/// `overrides` names it, so a test states the complete document it reads instead of patching a
/// template by text.
fn config<'a>(overrides: &'a [(&'a str, &'a str)]) -> String {
    let field = |name: &str, default: &'a str| -> &'a str {
        overrides
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(default, |(_, json)| *json)
    };
    format!(
        r#"{{"version": "2.0", "added_tokens": {}, "normalizer": {}, "pre_tokenizer": {},
            "post_processor": {}, "decoder": {}, "model": {}}}"#,
        field("added_tokens", "[]"),
        field("normalizer", "null"),
        field("pre_tokenizer", "null"),
        field("post_processor", "null"),
        field("decoder", "null"),
        field("model", TINY_BPE),
    )
}

fn read(text: &str) -> Result<PipelineTokenizer> {
    from_json(text)
}

/// The message from a config the reader must refuse. Not `unwrap_err`: `PipelineTokenizer` has
/// no `Debug`, which is what that would need on the `Ok` side.
fn read_err(text: &str) -> String {
    match read(text) {
        Ok(_) => panic!("expected the reader to refuse this config"),
        Err(e) => e.to_string(),
    }
}

fn ids(tok: &PipelineTokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, true)
        .wait()
        .unwrap()
        .iter()
        .flat_map(|e| e.ids())
        .map(|t| t.id())
        .collect()
}

#[test]
fn reads_a_tiny_bpe() {
    let tok = read(&config(&[])).unwrap();
    assert_eq!(ids(&tok, "abab"), vec![3]);
}

// ---- base64, the one piece of parsing that is not a field read ------------------------------

/// `crate::BASE64` is a spelled-out engine rather than `STANDARD`, purely so that decoding stays
/// indifferent to padding the way `spm_precompiled`'s `base64` 0.13 was. That is the only decision
/// left here, so it is the only thing worth a test -- swap the engine for `STANDARD` and this fails.
#[test]
fn a_charsmap_decodes_padded_or_not() {
    use base64::Engine as _;

    assert_eq!(crate::BASE64.decode("Zm9v").unwrap(), b"foo");
    assert_eq!(crate::BASE64.decode("Zg==").unwrap(), b"f");
    assert_eq!(crate::BASE64.decode("Zg").unwrap(), b"f", "unpadded");
    assert!(
        crate::BASE64.decode("Zm9v!").is_err(),
        "an illegal character"
    );
}

// ---- models ---------------------------------------------------------------------------------

#[test]
fn refuses_a_model_with_no_type() {
    // Inferring the kind from which keys are present is tk-convert's job, not this reader's.
    let json = config(&[(
        "model",
        r#"{"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]}"#,
    )]);
    assert!(read_err(&json).contains("`model` with no `type`"));
}


#[test]
fn refuses_legacy_string_merges() {
    // `"a b"` is the legacy spelling; tk-convert rewrites it to `["a", "b"]`.
    let legacy = config(&[(
        "model",
        r#"{"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
            "merges": ["a b", "ab ab"]}"#,
    )]);
    assert!(read_err(&legacy).contains("[left, right] pair"));
}



#[test]
#[cfg(feature = "unigram")]
fn reads_a_unigram_vocab_of_pairs() {
    let json = config(&[(
        "model",
        r#"{"type": "Unigram", "unk_id": 0, "byte_fallback": false,
            "vocab": [["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]]}"#,
    )]);
    // `ab` scores better than `a` + `a`, so the lattice must pick the pair.
    assert_eq!(ids(&read(&json).unwrap(), "ab"), vec![2]);
}

#[test]
#[cfg(feature = "unigram")]
fn refuses_a_unigram_entry_that_is_not_a_pair() {
    let json = config(&[("model", r#"{"type": "Unigram", "vocab": [["a"]]}"#)]);
    assert!(read_err(&json).contains("[token, score] pair"));
}

#[test]
#[cfg(feature = "wordpiece")]
fn reads_a_wordpiece_and_requires_every_field() {
    let full = config(&[(
        "model",
        r#"{"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "ab": 1, "@@c": 2}}"#,
    )]);
    assert_eq!(ids(&read(&full).unwrap(), "abc"), vec![1, 2]);

    // The config path's deserializer has no defaults either, so dropping a field must fail.
    let missing = full.replace(r#""max_input_chars_per_word": 100,"#, "");
    assert!(read_err(&missing).contains("max_input_chars_per_word"));
}

#[test]
#[cfg(feature = "wordlevel")]
fn reads_a_wordlevel() {
    let json = config(&[(
        "model",
        r#"{"type": "WordLevel", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "hello": 1}}"#,
    )]);
    assert_eq!(ids(&read(&json).unwrap(), "hello"), vec![1]);
}

// ---- normalizers ----------------------------------------------------------------------------

#[test]
fn flattens_a_normalizer_sequence_and_drops_an_empty_one() {
    let seq = config(&[(
        "normalizer",
        r#"{"type": "Sequence", "normalizers": [
            {"type": "Lowercase"},
            {"type": "Strip", "strip_left": true, "strip_right": true}
        ]}"#,
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

#[test]
fn strip_needs_both_sides_spelled_out() {
    let json = config(&[("normalizer", r#"{"type": "Strip", "strip_left": true}"#)]);
    assert!(read_err(&json).contains("strip_right"));
}

#[test]
#[cfg(feature = "normalizers")]
fn bert_normalizer_wants_strip_accents_present_even_as_null() {
    let ok = config(&[(
        "normalizer",
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
            "strip_accents": null, "lowercase": true}"#,
    )]);
    assert!(read(&ok).is_ok());

    let missing = config(&[(
        "normalizer",
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
            "lowercase": true}"#,
    )]);
    assert!(read_err(&missing).contains("strip_accents"));
}

#[test]
fn refuses_a_normalizer_it_does_not_know() {
    let json = config(&[("normalizer", r#"{"type": "Invented"}"#)]);
    assert!(read_err(&json).contains("`Invented` normalizer"));
}

// ---- pre-tokenizers -------------------------------------------------------------------------

#[test]
fn byte_level_without_use_regex_is_the_identity_split() {
    // The `Sequence[Split, ByteLevel]` idiom: the trailing ByteLevel only asks for the byte map,
    // which the model applies, so as a *splitter* it must be a no-op.
    let json = config(&[(
        "pre_tokenizer",
        r#"{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
            "use_regex": false}"#,
    )]);
    let doc = Json::parse(&json).unwrap();
    let (pretok, byte_level) = read_pre_tokenizer(doc.field("pre_tokenizer")).unwrap();
    assert!(matches!(pretok, PipelinePreTokenizer::None));
    // Still byte-level for the *model*, which is a separate switch.
    assert!(byte_level);
}

#[test]
fn byte_level_add_prefix_space_is_refused() {
    let json = config(&[(
        "pre_tokenizer",
        r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}"#,
    )]);
    assert!(read_err(&json).contains("add_prefix_space"));
}







// ---- post-processors ------------------------------------------------------------------------

#[test]
fn bert_and_roberta_processors_build_their_frames() {
    let bert = config(&[(
        "post_processor",
        r#"{"type": "BertProcessing", "sep": ["b", 1], "cls": ["a", 0]}"#,
    )]);
    // [CLS] $A [SEP] around the single sequence.
    assert_eq!(ids(&read(&bert).unwrap(), "abab"), vec![0, 3, 1]);

    let roberta = config(&[(
        "post_processor",
        r#"{"type": "RobertaProcessing", "sep": ["b", 1], "cls": ["a", 0],
            "trim_offsets": true, "add_prefix_space": true}"#,
    )]);
    assert_eq!(ids(&read(&roberta).unwrap(), "abab"), vec![0, 3, 1]);
}

#[test]
fn a_special_pair_must_carry_an_id() {
    let json = config(&[(
        "post_processor",
        r#"{"type": "BertProcessing", "sep": ["b"], "cls": ["a", 0]}"#,
    )]);
    assert!(read_err(&json).contains("[token, id] pair"));
}

// ---- decoders -------------------------------------------------------------------------------

#[test]
fn builds_every_decoder_variant() {
    // One config per variant, so a field rename in any of them fails here rather than silently
    // producing a decoder that decodes nothing.
    let cases = [
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
    ];
    for json in cases {
        let doc = Json::parse(json).unwrap();
        read_one_decoder(&doc).unwrap_or_else(|e| panic!("{json}: {e}"));
    }
}

#[test]
fn refuses_a_decoder_it_does_not_know() {
    let doc = Json::parse(r#"{"type": "Invented"}"#).unwrap();
    assert!(
        read_one_decoder(&doc)
            .unwrap_err()
            .to_string()
            .contains("`Invented` decoder")
    );
}

#[test]
fn a_decoder_reads_back_what_the_slim_path_wired_up() {
    let json = config(&[(
        "decoder",
        r#"{"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "Fuse"}
        ]}"#,
    )]);
    let tok = read(&json).unwrap();
    // `abab` is id 3; without a decoder the ids would join with a space instead.
    assert_eq!(tok.decode(&[3, 0], false).unwrap(), "ababa");
}

// ---- the real configs, when they are present ------------------------------------------------
/// A Unigram score is read as `serde_json`'s `f64`, not the correctly-rounded one.
///
/// `t5-base` spells `▁` as `-2.0122928619384766`. Correctly rounded that is `c000192d00000000`;
/// the vendored serde path lands one ULP away on `c000192d00000001`, and 8334 of t5's 32100 scores
/// are off by exactly that. Scores feed a Viterbi lattice, so reproducing serde bit for bit is what
/// keeps the ids that ship today -- see `crate::vendored`.
///
/// One ULP is also the *bound*: a real bug in the float path would miss by more, so this fails
/// either way -- if the deviation disappears, or if it grows.
#[test]
#[cfg(feature = "unigram")]
fn a_unigram_score_is_serdes_f64_one_ulp_off_the_correctly_rounded_one() {
    let json = config(&[(
        "model",
        r#"{"type": "Unigram", "unk_id": 0,
            "vocab": [["<unk>", 0.0], ["\u2581", -2.0122928619384766]]}"#,
    )]);
    let doc = Json::parse(&json).unwrap();
    let unigram = read_unigram(doc.field("model").unwrap()).unwrap();
    let (_, score) = unigram.iter().nth(1).expect("the second entry is `▁`");

    assert_eq!(
        score.to_bits(),
        0xc000_192d_0000_0001,
        "serde's f64, not another"
    );
    let correctly_rounded = f64::from(*score as f32);
    assert_eq!(
        (score.to_bits() as i64).abs_diff(correctly_rounded.to_bits() as i64),
        1,
        "the score must sit exactly one ULP from the f32 the file encodes"
    );
}
