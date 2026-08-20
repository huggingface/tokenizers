//! The reader's own tests: `TINY_BPE` plus one case per component it reads.

use super::decoders::read_one_decoder;
use super::normalizers::base64_decode;
use super::pre_tokenizers::read_prepend_scheme;
use super::*;
use tk_encode::pre_tokenizers::metaspace::PrependScheme;

/// A minimal BPE that needs no data files: two merges over a four-token vocab.
const TINY_BPE: &str = r#"{
    "version": "1.0",
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
        "merges": [["a", "b"], ["ab", "ab"]]
    }
}"#;

/// Swap `TINY_BPE`'s model out for another kind. Only the per-model tests need it.
#[cfg(any(feature = "unigram", feature = "wordpiece", feature = "wordlevel"))]
fn with_model(model: &str) -> String {
    TINY_BPE.replace(
        r#""model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
        "merges": [["a", "b"], ["ab", "ab"]]
    }"#,
        model,
    )
}

/// Swap one top-level component into `TINY_BPE`. The field is always spelled `null` there, so a
/// plain textual replace is unambiguous.
fn with_component(field: &str, json: &str) -> String {
    TINY_BPE.replace(
        &format!(r#""{field}": null"#),
        &format!(r#""{field}": {json}"#),
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
    let tok = read(TINY_BPE).unwrap();
    assert_eq!(ids(&tok, "abab"), vec![3]);
}

// ---- base64, the one piece of parsing that is not a field read ------------------------------

#[test]
fn base64_round_trips_every_tail_length() {
    // Reference vectors from RFC 4648 §10, which cover all three padding cases.
    for (encoded, decoded) in [
        ("", ""),
        ("Zg==", "f"),
        ("Zm8=", "fo"),
        ("Zm9v", "foo"),
        ("Zm9vYg==", "foob"),
        ("Zm9vYmE=", "fooba"),
        ("Zm9vYmFy", "foobar"),
    ] {
        assert_eq!(
            base64_decode(encoded).unwrap(),
            decoded.as_bytes(),
            "{encoded}"
        );
    }
}

#[test]
fn base64_decodes_without_padding_and_covers_the_alphabet() {
    // The same three bytes, padded and not: `spm_precompiled`'s own decoder is lenient here.
    assert_eq!(base64_decode("Zg").unwrap(), b"f");
    // `+` and `/` are the two non-alphanumeric symbols, and the ones a URL-safe alphabet moves.
    assert_eq!(base64_decode("++//").unwrap(), [0xfb, 0xef, 0xff]);
}

#[test]
fn base64_rejects_junk_and_truncation() {
    assert!(base64_decode("Zm9v!").is_err(), "an illegal character");
    assert!(base64_decode("Z").is_err(), "a lone trailing character");
    assert!(base64_decode("Zm9vZ").is_err(), "a truncated final group");
}

// ---- models ---------------------------------------------------------------------------------

#[test]
fn infers_the_model_kind_without_a_type_tag() {
    let cases = [
        (r#"{"merges": [], "vocab": {}}"#, "BPE"),
        (r#"{"vocab": [["a", 0.0]]}"#, "Unigram"),
        (
            r#"{"vocab": {}, "continuing_subword_prefix": "@@"}"#,
            "WordPiece",
        ),
        (r#"{"vocab": {}, "unk_token": "<unk>"}"#, "WordLevel"),
        (r#"{"type": "Unigram", "vocab": []}"#, "Unigram"),
        (r#"{"type": "Nonsense"}"#, "unknown"),
    ];
    for (json, want) in cases {
        let doc = Json::parse(json).unwrap();
        assert_eq!(model_kind(&doc), want, "{json}");
    }
}

#[test]
fn accepts_legacy_string_merges() {
    let legacy = TINY_BPE.replace(r#"[["a", "b"], ["ab", "ab"]]"#, r#"["a b", "ab ab"]"#);
    assert_eq!(ids(&read(&legacy).unwrap(), "abab"), vec![3]);
}

#[test]
fn refuses_a_merge_it_cannot_split() {
    let bad = TINY_BPE.replace(r#""a b""#, r#""ab""#);
    let bad = bad.replace(r#"[["a", "b"], ["ab", "ab"]]"#, r#"["ab"]"#);
    assert!(read_err(&bad).contains("no space"));
}

#[test]
#[cfg(feature = "unigram")]
fn reads_a_unigram_vocab_of_pairs() {
    let json = with_model(
        r#""model": {"type": "Unigram", "unk_id": 0, "byte_fallback": false,
            "vocab": [["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]]}"#,
    );
    // `ab` scores better than `a` + `a`, so the lattice must pick the pair.
    assert_eq!(ids(&read(&json).unwrap(), "ab"), vec![2]);
}

#[test]
#[cfg(feature = "unigram")]
fn refuses_a_unigram_entry_that_is_not_a_pair() {
    let json = with_model(r#""model": {"type": "Unigram", "vocab": [["a"]]}"#);
    assert!(read_err(&json).contains("[token, score] pair"));
}

#[test]
#[cfg(feature = "wordpiece")]
fn reads_a_wordpiece_and_requires_every_field() {
    let full = with_model(
        r#""model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "ab": 1, "@@c": 2}}"#,
    );
    assert_eq!(ids(&read(&full).unwrap(), "abc"), vec![1, 2]);

    // The config path's deserializer has no defaults either, so dropping a field must fail.
    let missing = full.replace(r#""max_input_chars_per_word": 100,"#, "");
    assert!(read_err(&missing).contains("max_input_chars_per_word"));
}

#[test]
#[cfg(feature = "wordlevel")]
fn reads_a_wordlevel() {
    let json = with_model(
        r#""model": {"type": "WordLevel", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "hello": 1}}"#,
    );
    assert_eq!(ids(&read(&json).unwrap(), "hello"), vec![1]);
}

// ---- normalizers ----------------------------------------------------------------------------

#[test]
fn flattens_a_normalizer_sequence_and_drops_an_empty_one() {
    let seq = with_component(
        "normalizer",
        r#"{"type": "Sequence", "normalizers": [
            {"type": "Lowercase"},
            {"type": "Strip", "strip_left": true, "strip_right": true}
        ]}"#,
    );
    let doc = Json::parse(&seq).unwrap();
    assert_eq!(
        read_normalizers(doc.get_some("normalizer")).unwrap().len(),
        2
    );

    let empty = with_component("normalizer", r#"{"type": "Sequence", "normalizers": []}"#);
    let doc = Json::parse(&empty).unwrap();
    assert!(
        read_normalizers(doc.get_some("normalizer"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn strip_needs_both_sides_spelled_out() {
    let json = with_component("normalizer", r#"{"type": "Strip", "strip_left": true}"#);
    assert!(read_err(&json).contains("strip_right"));
}

#[test]
#[cfg(feature = "normalizers")]
fn bert_normalizer_wants_strip_accents_present_even_as_null() {
    let ok = with_component(
        "normalizer",
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
            "strip_accents": null, "lowercase": true}"#,
    );
    assert!(read(&ok).is_ok());

    let missing = with_component(
        "normalizer",
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
            "lowercase": true}"#,
    );
    assert!(read_err(&missing).contains("strip_accents"));
}

#[test]
fn refuses_a_normalizer_it_does_not_know() {
    let json = with_component("normalizer", r#"{"type": "Invented"}"#);
    assert!(read_err(&json).contains("`Invented` normalizer"));
}

// ---- pre-tokenizers -------------------------------------------------------------------------

#[test]
fn byte_level_without_use_regex_is_the_identity_split() {
    // The `Sequence[Split, ByteLevel]` idiom: the trailing ByteLevel only asks for the byte map,
    // which the model applies, so as a *splitter* it must be a no-op.
    let json = with_component(
        "pre_tokenizer",
        r#"{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
            "use_regex": false}"#,
    );
    let doc = Json::parse(&json).unwrap();
    let mut normalizers = Vec::new();
    let (pretok, with_byte_level) =
        read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
    assert!(matches!(pretok, PipelinePreTokenizer::None));
    // Still byte-level for the *model*, which is a separate switch.
    assert!(with_byte_level);
}

#[test]
fn byte_level_add_prefix_space_is_refused() {
    let json = with_component(
        "pre_tokenizer",
        r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}"#,
    );
    assert!(read_err(&json).contains("add_prefix_space"));
}

#[test]
fn metaspace_becomes_a_normalizer_plus_a_split() {
    let json = with_component(
        "pre_tokenizer",
        r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
    );
    let doc = Json::parse(&json).unwrap();
    let mut normalizers = Vec::new();
    let (pretok, with_byte_level) =
        read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
    assert!(!with_byte_level);
    assert!(matches!(pretok, PipelinePreTokenizer::Split(_)));
    assert!(matches!(
        normalizers.as_slice(),
        [PipelineNormalizer::Metaspace(_)]
    ));
}

#[test]
fn t5_shape_collapses_to_one_split_not_a_sequence() {
    let json = with_component(
        "pre_tokenizer",
        r#"{"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
        ]}"#,
    );
    let doc = Json::parse(&json).unwrap();
    let mut normalizers = Vec::new();
    let (pretok, _) = read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
    // A `Sequence` here would run the whitespace split again over already-marked text.
    assert!(matches!(pretok, PipelinePreTokenizer::Split(_)));
    assert_eq!(normalizers.len(), 1);
}

#[test]
fn the_metaspace_normalizer_lands_after_the_declared_one() {
    let json = with_component("normalizer", r#"{"type": "Lowercase"}"#);
    let json = json.replace(
        r#""pre_tokenizer": null"#,
        r#""pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
    );
    let doc = Json::parse(&json).unwrap();
    let mut normalizers = read_normalizers(doc.get_some("normalizer")).unwrap();
    read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
    // The config asks for the whole normalizer first, then the pre-tokenizer.
    assert!(matches!(
        normalizers.as_slice(),
        [
            PipelineNormalizer::Lowercase(_),
            PipelineNormalizer::Metaspace(_)
        ]
    ));
}

#[test]
fn prepend_scheme_reproduces_the_config_paths_rule() {
    let parse = |json: &str| {
        let doc = Json::parse(json).unwrap();
        read_prepend_scheme(&doc)
    };
    // Neither key: the default.
    assert_eq!(parse("{}").unwrap(), PrependScheme::Always);
    // The old key alone, which is what t5 and albert ship.
    assert_eq!(
        parse(r#"{"add_prefix_space": true}"#).unwrap(),
        PrependScheme::Always
    );
    // `add_prefix_space: true` is ignored outright, so an explicit scheme wins over it — even a
    // contradicting one.
    assert_eq!(
        parse(r#"{"add_prefix_space": true, "prepend_scheme": "never"}"#).unwrap(),
        PrependScheme::Never
    );
    // And `false` is checked against the *defaulted* scheme, which is `Always`. So the old key
    // alone can never spell `false`, and `false` is only accepted next to the `never` it would
    // have set. Both quirks are the config path's, reproduced because ids depend on them.
    assert!(parse(r#"{"add_prefix_space": false}"#).is_err());
    assert!(parse(r#"{"add_prefix_space": false, "prepend_scheme": "always"}"#).is_err());
    assert_eq!(
        parse(r#"{"add_prefix_space": false, "prepend_scheme": "never"}"#).unwrap(),
        PrependScheme::Never
    );
    assert!(parse(r#"{"prepend_scheme": "sometimes"}"#).is_err());
}

#[test]
fn refuses_the_metaspace_settings_it_cannot_rebuild() {
    for (why, pretok) in [
        (
            "split: false",
            r#"{"type": "Metaspace", "replacement": "▁", "split": false}"#,
        ),
        (
            "prepend_scheme: first",
            r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}"#,
        ),
        (
            "a metaspace buried in a sequence",
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "Whitespace"},
                {"type": "Metaspace", "replacement": "▁"}
            ]}"#,
        ),
    ] {
        let json = with_component("pre_tokenizer", pretok);
        assert!(read(&json).is_err(), "{why}");
    }
}

#[test]
fn a_multi_character_replacement_is_not_truncated() {
    let json = with_component(
        "pre_tokenizer",
        r#"{"type": "Metaspace", "replacement": "ab"}"#,
    );
    assert!(read_err(&json).contains("exactly one character"));
}

// ---- post-processors ------------------------------------------------------------------------

#[test]
fn bert_and_roberta_processors_build_their_frames() {
    let bert = with_component(
        "post_processor",
        r#"{"type": "BertProcessing", "sep": ["b", 1], "cls": ["a", 0]}"#,
    );
    // [CLS] $A [SEP] around the single sequence.
    assert_eq!(ids(&read(&bert).unwrap(), "abab"), vec![0, 3, 1]);

    let roberta = with_component(
        "post_processor",
        r#"{"type": "RobertaProcessing", "sep": ["b", 1], "cls": ["a", 0],
            "trim_offsets": true, "add_prefix_space": true}"#,
    );
    assert_eq!(ids(&read(&roberta).unwrap(), "abab"), vec![0, 3, 1]);
}

#[test]
fn a_special_pair_must_carry_an_id() {
    let json = with_component(
        "post_processor",
        r#"{"type": "BertProcessing", "sep": ["b"], "cls": ["a", 0]}"#,
    );
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
        r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
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
    let json = with_component(
        "decoder",
        r#"{"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "Fuse"}
        ]}"#,
    );
    let tok = read(&json).unwrap();
    // `abab` is id 3; without a decoder the ids would join with a space instead.
    assert_eq!(tok.decode(&[3, 0], false).unwrap(), "ababa");
}

// ---- the real configs, when they are present ------------------------------------------------

/// t5's Unigram scores, which are the reason this reader emulates `serde_json`'s float
/// arithmetic instead of using `f64::from_str`.
///
/// A SentencePiece vocabulary is trained in `f32`, so every score in the file is exactly an
/// `f32` widened to `f64`, and `f64::from_str` lands on it. `serde_json` (without
/// `float_roundtrip`, which is off by default) misses 8,334 of t5's 32,100 by one ULP. Scores
/// feed a Viterbi lattice, so that flips a near-tie roughly twice per 1.25M tokens.
///
/// We reproduce serde rather than improve on it, because the ids that ship today are the
/// contract. So this pins two things: the scores are bit-identical to the config path (which is
/// what makes t5 byte-exact in `json_oracle`), and they are deliberately *not* all the
/// correctly-rounded `f32` values — with any deviation bounded to one ULP, so a real parsing
/// bug could not hide behind this allowance.
/// `json.rs` reads Unigram scores as `f64` from decimal text; `serde_json` reads
/// the same text and rounds identically. That equality is pinned directly against `serde_json`
/// in `json.rs` (`matches_serde_not_from_str_on_a_real_unigram_score` and
/// `numbers_are_bit_identical_to_serde_json`), so what is left to check here is the *bound* on
/// the error over a whole real vocabulary: every score is within one ULP of the `f32` the file
/// actually encodes, and at least one is not exactly it -- which is the cost we knowingly accept
/// for not pulling in `serde_json/float_roundtrip`.
#[test]
#[cfg(feature = "unigram")]
fn unigram_scores_stay_within_one_ulp_of_the_f32_the_file_encodes() {
    let path = "../data/t5-base.json";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let text = std::fs::read_to_string(path).unwrap();
    let doc = Json::parse(&text).unwrap();
    let slim = read_unigram(doc.get_some("model").unwrap()).unwrap();

    let mut off_by_one_ulp = 0usize;
    for (tok, score) in slim.iter() {
        let correctly_rounded = f64::from(*score as f32);
        if score.to_bits() != correctly_rounded.to_bits() {
            let delta = (score.to_bits() as i64).abs_diff(correctly_rounded.to_bits() as i64);
            assert!(
                delta <= 1,
                "score for {tok:?} is {delta} ULP from the f32 the file encodes, which is more \
                 than serde's rounding can explain"
            );
            off_by_one_ulp += 1;
        }
    }
    assert!(
        off_by_one_ulp > 0,
        "every score is now correctly rounded, so the parser's float path must have changed: \
         re-check it against `serde_json` before relaxing this test"
    );
}
