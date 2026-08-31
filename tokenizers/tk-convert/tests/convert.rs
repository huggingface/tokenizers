//! The canonicalisation rules, and the gate that proves the point: after this pass, the canonical
//! reader can read the file.

use serde_json::{Map, Value};
use tk_convert::{ConvertError, canonicalize_file, canonicalize_str, canonicalize_value};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

/// The smallest config that is a valid `tokenizer.json`, around `model` plus any extra slots.
fn config(model: &str, extra: &str) -> Value {
    serde_json::from_str(&format!(r#"{{"version": "1.0", "model": {model}{extra}}}"#))
        .expect("the test literal is not JSON")
}

fn done(model: &str, extra: &str) -> Value {
    let mut v = config(model, extra);
    canonicalize_value(&mut v).expect("canonicalisation failed");
    assert_no_legacy_residue(&v, "the hand-written literal");
    v
}

/// The message a config is refused with.
fn err(model: &str, extra: &str) -> String {
    let mut v = config(model, extra);
    canonicalize_value(&mut v)
        .expect_err("expected a refusal")
        .to_string()
}

const UNIGRAM: &str = r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#;

const BPE: &str = r#"{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]}"#;

/// Not one legacy shape survived. Checkable on the JSON alone, so it does not depend on what the
/// reader happens to tolerate -- which is what lets the reader keep no legacy branch at all.
fn assert_no_legacy_residue(v: &Value, what: &str) {
    assert_eq!(
        v.get("version").and_then(Value::as_str),
        Some("2.0"),
        "{what}: version"
    );
    let model = v["model"].as_object().expect("no model object");
    assert!(
        model.contains_key("type"),
        "{what}: the model has no `type`"
    );
    assert!(
        !model.contains_key("files"),
        "{what}: the vocabulary is a path"
    );
    assert!(!model["vocab"].is_string(), "{what}: `vocab` is a path");
    if model.get("type").and_then(Value::as_str) == Some("BPE") {
        assert!(
            model.contains_key("byte_level"),
            "{what}: BPE without `byte_level`"
        );
    }
    for m in model
        .get("merges")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        assert!(!m.is_string(), "{what}: a merge is still space-joined");
    }
    // Both are two components in disguise; the canonical file spells them as what they are.
    walk(&v["pre_tokenizer"], &mut |o| {
        let tag = o.get("type").and_then(Value::as_str);
        assert!(
            tag != Some("Metaspace") && tag != Some("ByteLevel"),
            "{what}: a {tag:?} survives in the pre_tokenizer slot"
        );
    });
    // What is left of a `Metaspace` is a decoder, which the writer spells without the legacy keys.
    walk(v, &mut |o| {
        if o.get("type").and_then(Value::as_str) == Some("Metaspace") {
            assert!(
                o.contains_key("prepend_scheme"),
                "{what}: Metaspace without a scheme"
            );
            for dead in ["add_prefix_space", "str_rep", "split"] {
                assert!(
                    !o.contains_key(dead),
                    "{what}: a Metaspace still has `{dead}`"
                );
            }
        }
    });
}

fn walk(node: &Value, f: &mut impl FnMut(&Map<String, Value>)) {
    match node {
        Value::Object(o) => {
            f(o);
            o.values().for_each(|v| walk(v, f));
        }
        Value::Array(items) => items.iter().for_each(|v| walk(v, f)),
        _ => {}
    }
}

// ---- the model ------------------------------------------------------------------------------

#[test]
fn infers_a_missing_model_type() {
    // `merges` **must** beat `continuing_subword_prefix`: a serialized BPE writes that key as
    // null, so gpt2 would otherwise read as a WordPiece with no merges.
    for (model, want) in [
        (
            r#"{"vocab": {}, "merges": [], "continuing_subword_prefix": null}"#,
            "BPE",
        ),
        (
            r###"{"vocab": {}, "continuing_subword_prefix": "##"}"###,
            "WordPiece",
        ),
        (r#"{"vocab": [["a", 0.0]], "unk_id": 0}"#, "Unigram"),
        (r#"{"vocab": {}}"#, "WordLevel"),
        (
            r#"{"type": "WordLevel", "vocab": {}, "merges": []}"#,
            "WordLevel",
        ),
    ] {
        assert_eq!(done(model, "")["model"]["type"], want, "{model}");
    }
}

#[test]
fn rewrites_legacy_merges_into_pairs() {
    let pairs = |m: &str| done(m, "")["model"]["merges"].clone();
    // The `merges.txt` header is dropped, not rewritten: it is not a merge.
    assert_eq!(
        pairs(r##"{"vocab": {}, "merges": ["#version: 0.2", "a b"]}"##),
        serde_json::json!([["a", "b"]])
    );
    assert_eq!(
        pairs(r#"{"vocab": {}, "merges": [["a", "b"]]}"#),
        serde_json::json!([["a", "b"]])
    );
    // A token containing a space has never loaded, so guessing where to split would invent a
    // different tokenizer rather than report one.
    let mut bad = config(r#"{"vocab": {}, "merges": ["a b c"]}"#, "");
    assert!(matches!(
        canonicalize_value(&mut bad).unwrap_err(),
        ConvertError::BadMerge { .. }
    ));
}

// ---- Metaspace ------------------------------------------------------------------------------

#[test]
fn metaspace_fields_follow_the_legacy_rules() {
    let ms = |fields: &str| format!(r#", "decoder": {{"type": "Metaspace"{fields}}}"#);
    let scheme = |fields: &str| done(BPE, &ms(fields))["decoder"]["prepend_scheme"].clone();

    // Absent means `always`: the old `add_prefix_space` defaulted to true.
    assert_eq!(scheme(r#", "replacement": "▁""#), "always");
    // `add_prefix_space: true` is ignored outright, so an explicit scheme always wins.
    assert_eq!(
        scheme(r#", "replacement": "▁", "add_prefix_space": true, "prepend_scheme": "never""#),
        "never"
    );
    // `false` is checked against the *defaulted* scheme, so it needs an agreeing `never`.
    assert_eq!(
        scheme(r#", "replacement": "▁", "add_prefix_space": false, "prepend_scheme": "never""#),
        "never"
    );
    assert!(
        err(
            BPE,
            &ms(r#", "replacement": "▁", "add_prefix_space": false"#)
        )
        .contains("does not match")
    );
    assert!(
        err(
            BPE,
            &ms(r#", "replacement": "▁", "prepend_scheme": "sometimes""#)
        )
        .contains("unknown metaspace prepend_scheme")
    );
    // Checked, not truncated.
    assert!(err(BPE, &ms(r#", "replacement": "ab""#)).contains("exactly one character"));
    assert!(err(BPE, &ms("")).contains("no `replacement`"));
}

#[test]
fn a_metaspace_pre_tokenizer_becomes_a_normalizer_and_a_split() {
    // Lone: the delimiter half goes to the normalizer slot, the cut stays a `Split`.
    let v = done(
        BPE,
        r#", "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}"#,
    );
    assert_eq!(
        v["normalizer"],
        serde_json::json!({"type": "MetaspaceNormalizer", "replacement": "▁",
                           "prepend": "always", "drop_whitespace": false})
    );
    assert_eq!(
        v["pre_tokenizer"],
        serde_json::json!({"type": "Split", "pattern": {"String": "▁"},
                           "behavior": "MergedWithNext", "invert": false})
    );

    // All three schemes carry through by name. `first` is the pre-Tekken Mistral spelling, and it
    // survives because the normalizer is told whether its text opens the sequence.
    for scheme in ["always", "first", "never"] {
        let v = done(
            BPE,
            &format!(
                r#", "pre_tokenizer": {{"type": "Metaspace", "replacement": "▁",
                     "prepend_scheme": "{scheme}"}}"#
            ),
        );
        assert_eq!(v["normalizer"]["prepend"], scheme);
    }

    // The t5/albert pair: the `WhitespaceSplit` was never a component, it was `drop_whitespace`.
    let v = done(
        BPE,
        r#", "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
             {"type": "WhitespaceSplit"}, {"type": "Metaspace", "replacement": "▁"}]}"#,
    );
    assert_eq!(v["normalizer"]["drop_whitespace"], true);
    assert_eq!(v["pre_tokenizer"]["type"], "Split");

    // That pair only works under `always`. With the whitespace gone the delimiter is the only cut
    // left, so under `first` or `never` the words run together into one span where the released
    // crate keeps one per word. Refused rather than converted into different ids.
    for scheme in ["first", "never"] {
        let e = err(
            BPE,
            &format!(
                r#", "pre_tokenizer": {{"type": "Sequence", "pretokenizers": [
                     {{"type": "WhitespaceSplit"}},
                     {{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "{scheme}"}}]}}"#
            ),
        );
        assert!(
            e.contains(&format!("with `prepend_scheme: {scheme}` is not supported")),
            "{scheme}: {e}"
        );
    }

    // The delimiter half lands *after* a declared normalizer -- the order the old reader applied,
    // and the one the added-token matcher depends on.
    let v = done(
        BPE,
        r#", "normalizer": {"type": "NFKC"}
           , "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}"#,
    );
    let chain = &v["normalizer"]["normalizers"];
    assert_eq!(chain[0]["type"], "NFKC");
    assert_eq!(chain[1]["type"], "MetaspaceNormalizer");

    // `split: false` wrote delimiters but never cut, and the pair has no way to say that.
    assert!(
        err(
            BPE,
            r#", "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "split": false}"#
        )
        .contains("split: false")
    );
}

// ---- ByteLevel ------------------------------------------------------------------------------

#[test]
fn a_byte_level_pre_tokenizer_becomes_a_model_flag_and_a_split() {
    // `use_regex` defaults to true: the split it asked for is the GPT-2 regex.
    let v = done(BPE, r#", "pre_tokenizer": {"type": "ByteLevel"}"#);
    let pretok = &v["pre_tokenizer"];
    assert_eq!(v["model"]["byte_level"], true);
    assert_eq!(pretok["pattern"]["Regex"], bitsplit::regexes::GPT2);

    // `use_regex: false` asked only for the byte map, so the member simply goes.
    let v = done(
        BPE,
        r#", "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
             {"type": "WhitespaceSplit"}, {"type": "ByteLevel", "use_regex": false}]}"#,
    );
    assert_eq!(v["model"]["byte_level"], true);
    assert_eq!(
        v["pre_tokenizer"]["pretokenizers"],
        serde_json::json!([{"type": "WhitespaceSplit"}])
    );

    // A model that never had one still has to say so.
    assert_eq!(done(BPE, "")["model"]["byte_level"], false);

    // Guards the canonical reader no longer carries: it reads `byte_level` only off a BPE, so
    // one anywhere else would be silently ignored rather than refused.
    let bl = r#", "pre_tokenizer": {"type": "ByteLevel"}"#;
    assert!(
        err(
            BPE,
            r#", "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true}"#
        )
        .contains("add_prefix_space")
    );
    assert!(
        err(
            BPE,
            r#", "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
             {"type": "ByteLevel"}, {"type": "WhitespaceSplit"}]}"#
        )
        .contains("last member")
    );
    assert!(err(UNIGRAM, bl).contains("needs a BPE model"));
}

// ---- the post-processor ------------------------------------------------------------------------

#[test]
fn a_template_states_its_ids_instead_of_naming_them() {
    let v = done(
        BPE,
        r#", "post_processor": {"type": "TemplateProcessing",
             "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}}],
             "pair": [],
             "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]}}}"#,
    );
    let pp = &v["post_processor"];
    assert_eq!(
        pp["single"],
        serde_json::json!([{"ids": [2], "type_id": 0}, {"seq": "A", "type_id": 0}])
    );
    // The lookup table was the reason for the names, and both go.
    assert!(pp.get("special_tokens").is_none());
}

#[test]
fn data_objects_are_not_mistaken_for_components() {
    // No `replacement`, so if the component walk reached this it would be
    // `MetaspaceNoReplacement`. It does not: the table is data, and the template lowering
    // consumes it rather than descending into it.
    let v = done(
        BPE,
        r#", "post_processor": {"type": "TemplateProcessing", "single": [], "pair": [],
             "special_tokens": {"x": {"type": "Metaspace"}}}"#,
    );
    assert!(v["post_processor"].get("special_tokens").is_none());
}

// ---- the pass itself -------------------------------------------------------------------------

#[test]
fn refuses_something_that_is_not_a_tokenizer_config() {
    use ConvertError::*;
    assert!(matches!(canonicalize_str("[]"), Err(NotAnObject { .. })));
    assert!(matches!(canonicalize_str("{}"), Err(MissingModel)));
    assert!(matches!(
        canonicalize_str(r#"{"model": 3}"#),
        Err(ModelNotObject { .. })
    ));
    assert!(matches!(canonicalize_str("not json"), Err(Json(_))));
}

/// Safe to run unconditionally in front of a reader, which means running it twice must change
/// nothing the first pass did not.
#[test]
fn is_idempotent() {
    let legacy = r##"{"version": "1.0",
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": "▁", "str_rep": "▁", "add_prefix_space": true}]},
        "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true},
        "model": {"continuing_subword_prefix": null, "vocab": {"a": 0, "b": 1, "ab": 2},
                  "merges": ["#version: 0.2", "a b"]}}"##;
    let once = canonicalize_str(legacy).unwrap();
    assert_eq!(
        once,
        canonicalize_str(&once).unwrap(),
        "the pass is not idempotent"
    );
    assert_no_legacy_residue(&serde_json::from_str(&once).unwrap(), "the legacy literal");
}

// ---- end to end ------------------------------------------------------------------------------

/// The test that proves the point of the module: after this pass the *canonical* reader can read
/// the file. `tk_serialize` is a dev-dependency with every component feature on, so this cannot
/// silently drop to "read nothing and pass".
///
/// The contract is absolute -- every fixture converts and reads, or is named in `UNCONVERTIBLE`.
/// Comparing the reader's verdict before and after would say nothing now: the reader refuses every
/// raw `1.0` file, so "before" is the same version error every time.
#[test]
fn every_fixture_canonicalises_into_something_the_canonical_reader_accepts() {
    /// Fixtures with no canonical form, each with a substring its refusal must contain. Every
    /// entry is a limit of what the pipeline can *build*, not a gap in this pass; listing them
    /// makes adding one a diff a reviewer sees. Checked both ways, so a stale entry fails too.
    const UNCONVERTIBLE: &[(&str, &str)] = &[("tokenizer.json", "add_prefix_space")];

    let Ok(entries) = std::fs::read_dir(std::path::Path::new(DATA)) else {
        eprintln!("skipping: no fixtures at {DATA} (populated by `make data models`)");
        return;
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();

    let (mut read, mut unconvertible, mut skipped) = (0usize, 0usize, 0usize);
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(path).unwrap();
        // `gpt2-vocab.json` is a bare {token: id} map and `unigram.json` a bare model: neither is
        // a tokenizer.json.
        if !serde_json::from_str::<Value>(&text)
            .is_ok_and(|v| v.get("model").is_some_and(Value::is_object))
        {
            skipped += 1;
            continue;
        }
        let listed = UNCONVERTIBLE.iter().find(|(n, _)| *n == name);
        let canonical = match canonicalize_file(path) {
            Err(e) => {
                let (_, why) = listed
                    .unwrap_or_else(|| panic!("{name}: this pass refuses to convert it: {e}"));
                assert!(
                    e.to_string().contains(why),
                    "{name}: not the listed reason: {e}"
                );
                unconvertible += 1;
                continue;
            }
            Ok(c) => {
                assert!(
                    listed.is_none(),
                    "{name}: listed as unconvertible, but it converted"
                );
                c
            }
        };
        assert_no_legacy_residue(&serde_json::from_str(&canonical).unwrap(), &name);
        if let Err(e) = tk_serialize::from_json(&canonical) {
            panic!("{name}: canonicalised, but the canonical reader still refuses it: {e}");
        }
        read += 1;
    }
    eprintln!(
        "{read} converted and read; {unconvertible} with no canonical form; {skipped} not configs"
    );
    assert!(read > 0, "no fixture was actually read");
}
