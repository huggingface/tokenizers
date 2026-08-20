//! The reader's own tests: what the whole-config gates cannot reach.
//!
//! `agrees_with_serde_json_on_every_real_config` in [`crate::json`] and
//! `round_trip_preserves_ids_on_every_real_config` in [`crate::to_json`] already run every
//! `data/*.json` through this reader end to end, and `tk-convert`'s oracles compare the ids and the
//! decoded text against an independent implementation. So a test earns its place here only if it
//! covers something none of those can: an **error path**, which no valid config exercises, or a
//! **shape absent from `data/`**, which no fixture contains.

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

/// The `model` object of [`TINY_BPE`] verbatim, so a test can swap it for another kind.
const TINY_BPE_MODEL: &str = r#"{
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
        "merges": [["a", "b"], ["ab", "ab"]]
    }"#;

/// Swap one top-level component into `TINY_BPE`. `model` holds the tiny BPE verbatim and every
/// other such field is spelled `null` there, so a plain textual replace is unambiguous either way.
fn with_field(field: &str, json: &str) -> String {
    if field == "model" {
        return TINY_BPE.replace(TINY_BPE_MODEL, json);
    }
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

/// Only [`reads_a_wordlevel`] encodes anything here: every other test on this side either reads a
/// component apart from the pipeline or checks a refusal, so this is gated with its one caller.
#[cfg(feature = "wordlevel")]
fn ids(tok: &PipelineTokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, true)
        .wait()
        .unwrap()
        .iter()
        .flat_map(|e| e.ids())
        .map(|t| t.id())
        .collect()
}

// ---- base64, the one piece of parsing that is not a field read ------------------------------

/// The decoder is ours and hand-written, so it keeps a gate: what it must refuse, and the one
/// leniency it must keep. The round trip over every tail length lives with the encoder it has to
/// agree with, in `to_json::normalizers`.
#[test]
fn base64_rejects_junk_but_tolerates_missing_padding() {
    for (why, encoded) in [
        ("an illegal character", "Zm9v!"),
        ("a lone trailing character", "Z"),
        ("a truncated final group", "Zm9vZ"),
    ] {
        assert!(base64_decode(encoded).is_err(), "{why}");
    }
    // The same three bytes, padded and not: `spm_precompiled`'s own decoder is lenient here.
    assert_eq!(base64_decode("Zg").unwrap(), b"f");
    // `+` and `/` are the two non-alphanumeric symbols, and the ones a URL-safe alphabet moves.
    assert_eq!(base64_decode("++//").unwrap(), [0xfb, 0xef, 0xff]);
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

/// `WordLevel` is the one model kind no `data/` fixture holds, so this is the only thing that reads
/// the path at all.
#[test]
#[cfg(feature = "wordlevel")]
fn reads_a_wordlevel() {
    let json = with_field(
        "model",
        r#"{"type": "WordLevel", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "hello": 1}}"#,
    );
    assert_eq!(ids(&read(&json).unwrap(), "hello"), vec![1]);
}

// ---- pre-tokenizers -------------------------------------------------------------------------

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

// ---- the error paths, which no valid config reaches -----------------------------------------

/// Every refusal the reader owns, as one table of (built in this configuration, field, config, what
/// the message must name).
///
/// One test rather than one per refusal: each is a one-line contract — this input, that message —
/// and what matters is that the list is complete, which a table shows and a dozen functions hide.
/// The message is asserted rather than just the failure, because "refuses everything" would pass a
/// test that only checked `is_err`.
///
/// The three model and normalizer kinds behind a feature carry a `cfg!` flag rather than a `#[cfg]`
/// attribute: with the feature off the reader refuses them for a different reason — the component is
/// not compiled in — so the entry has to be skipped, and a runtime flag keeps the whole list visible
/// in every build instead of hiding a third of it.
#[test]
fn refuses_what_it_cannot_rebuild() {
    for (built, field, json, expected) in [
        // A legacy merge with nothing to split on. The pair spelling is unambiguous; this one is
        // not, so it is `tk-convert`'s to resolve rather than this reader's to guess.
        (
            true,
            "model",
            r#"{"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["ab"]}"#,
            "no space",
        ),
        // Both sides of a `Strip` have to be spelled out: the config path's deserializer has no
        // default for them either.
        (
            true,
            "normalizer",
            r#"{"type": "Strip", "strip_left": true}"#,
            "strip_right",
        ),
        (
            true,
            "normalizer",
            r#"{"type": "Invented"}"#,
            "`Invented` normalizer",
        ),
        (
            true,
            "decoder",
            r#"{"type": "Invented"}"#,
            "`Invented` decoder",
        ),
        // A `ByteLevel` that prepends a space is a text rewrite, and the pipeline has no
        // normalizer to hang it on.
        (
            true,
            "pre_tokenizer",
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}"#,
            "add_prefix_space",
        ),
        // The three `Metaspace` settings that cannot be rebuilt as a normalizer plus a split: a
        // `Metaspace` that does not split, one that prepends only at the start of the text (a
        // normalizer sees one chunk at a time and never knows), and one buried in a `Sequence`
        // anywhere other than after a `WhitespaceSplit`.
        (
            true,
            "pre_tokenizer",
            r#"{"type": "Metaspace", "replacement": "▁", "split": false}"#,
            "split: false",
        ),
        (
            true,
            "pre_tokenizer",
            r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}"#,
            "prepend_scheme: first",
        ),
        (
            true,
            "pre_tokenizer",
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "Whitespace"},
                {"type": "Metaspace", "replacement": "▁"}
            ]}"#,
            "other than on its own",
        ),
        // A replacement is one character, and a longer one must not be silently truncated to it.
        (
            true,
            "pre_tokenizer",
            r#"{"type": "Metaspace", "replacement": "ab"}"#,
            "exactly one character",
        ),
        // The five things a template must get right, which the released library enforces. None of
        // these moves an id on any `data/` fixture -- no fixture trips them -- but a `tokenizer.json`
        // is untrusted input, and a template that names no sequence, or names one twice, produces a
        // silently wrong encoding rather than a refusal. `type_id` used to be truncated with
        // `as u8`, so `256` became `0`.
        (
            true,
            "post_processor",
            r#"{"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 256}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {}}"#,
            "does not fit a u8",
        ),
        (
            true,
            "post_processor",
            r#"{"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {}}"#,
            "references sequence A 2 times",
        ),
        (
            true,
            "post_processor",
            r#"{"type": "TemplateProcessing", "single": [], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {}}"#,
            "references sequence A 0 times",
        ),
        (
            true,
            "post_processor",
            r#"{"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {}}"#,
            "`single` template references sequence B 1 times",
        ),
        (
            true,
            "post_processor",
            r#"{"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}], "special_tokens": {}}"#,
            "`pair` template references sequence B 0 times",
        ),
        // A frame's special token is `[token, id]`, and the id is what the frame is made of.
        (
            true,
            "post_processor",
            r#"{"type": "BertProcessing", "sep": ["b"], "cls": ["a", 0]}"#,
            "[token, id] pair",
        ),
        // A Unigram entry that is not a `[token, score]` pair.
        (
            cfg!(feature = "unigram"),
            "model",
            r#"{"type": "Unigram", "vocab": [["a"]]}"#,
            "[token, score] pair",
        ),
        // WordPiece has no defaults on the config path either, so a dropped field must fail rather
        // than be filled in.
        (
            cfg!(feature = "wordpiece"),
            "model",
            r#"{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "@@",
            "vocab": {"[UNK]": 0, "ab": 1}}"#,
            "max_input_chars_per_word",
        ),
        // `strip_accents: null` is not "absent": it means "decide from `lowercase`", and the key has
        // to be there for exactly that reason.
        (
            cfg!(feature = "normalizers"),
            "normalizer",
            r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
            "lowercase": true}"#,
            "strip_accents",
        ),
    ] {
        if !built {
            continue;
        }
        let error = read_err(&with_field(field, json));
        assert!(
            error.contains(expected),
            "{field} = {json}\n  was refused with {error:?}, which does not name {expected:?}"
        );
    }
}
