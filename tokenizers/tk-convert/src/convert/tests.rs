//! Tests for the canonicalisation pass.
use super::*;

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

const BPE: &str = r#"{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]}"#;

/// Not one legacy shape survived. Checkable on the JSON alone, so it does not depend on what the
/// reader happens to tolerate -- which is what lets the reader keep no legacy branch at all.
fn assert_no_legacy_residue(v: &Value, what: &str) {
    let say = |cond: bool, why: &str| assert!(cond, "{what}: {why}");
    say(
        v.get("version").and_then(Value::as_str) == Some(VERSION),
        "the version is not 2.0",
    );

    let model = v["model"].as_object().expect("no model object");
    say(model.contains_key("type"), "the model still has no `type`");
    say(!model.contains_key("files"), "the vocabulary is still a path");
    say(
        !model.get("vocab").is_some_and(Value::is_string),
        "`vocab` is still a path",
    );
    say(
        model.get("type").and_then(Value::as_str) != Some("BPE")
            || model.get("byte_level").is_some(),
        "the BPE model does not state `byte_level`",
    );
    for m in model.get("merges").and_then(Value::as_array).unwrap_or(&vec![]) {
        say(!m.is_string(), "a merge is still a space-joined string");
    }

    // Both are two components in disguise; the canonical file spells them as what they are.
    walk(&v["pre_tokenizer"], &mut |o| {
        let tag = o.get("type").and_then(Value::as_str);
        say(
            tag != Some("Metaspace") && tag != Some("ByteLevel"),
            "a `Metaspace` or `ByteLevel` survives in the pre_tokenizer slot",
        );
    });
    // What is left of a `Metaspace` is a decoder, which the writer spells with neither the legacy
    // keys nor `split`.
    walk(v, &mut |o| {
        if o.get("type").and_then(Value::as_str) == Some("Metaspace") {
            for dead in ["add_prefix_space", "str_rep", "split"] {
                say(!o.contains_key(dead), &format!("a Metaspace still has `{dead}`"));
            }
            say(
                o.contains_key("prepend_scheme"),
                "a Metaspace has no `prepend_scheme`",
            );
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
        (r#"{"vocab": {}, "merges": [], "continuing_subword_prefix": null}"#, "BPE"),
        (r###"{"vocab": {}, "continuing_subword_prefix": "##"}"###, "WordPiece"),
        (r#"{"vocab": [["a", 0.0]], "unk_id": 0}"#, "Unigram"),
        (r#"{"vocab": {}}"#, "WordLevel"),
        (r#"{"type": "WordLevel", "vocab": {}, "merges": []}"#, "WordLevel"),
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

/// A `Metaspace` in the slot one still lives in. The pre-tokenizer form is two components and is
/// lowered away; the decoder keeps the tag, so it is where the field rules stay observable.
fn decoder(spelling: &str) -> Result<Value, ConvertError> {
    let mut v = config(BPE, &format!(r#", "decoder": {spelling}"#));
    canonicalize_value(&mut v)?;
    Ok(v["decoder"].clone())
}

#[test]
fn metaspace_fields_follow_the_legacy_rules() {
    let scheme = |s: &str| decoder(s).map(|d| d["prepend_scheme"].clone());
    // Absent means `always`, because the old `add_prefix_space` defaulted to true.
    assert_eq!(scheme(r#"{"type": "Metaspace", "replacement": "▁"}"#).unwrap(), "always");
    // `add_prefix_space: true` is ignored outright, so an explicit scheme always wins.
    assert_eq!(
        scheme(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true,
                   "prepend_scheme": "never"}"#).unwrap(),
        "never"
    );
    // `false` is checked against the *defaulted* scheme, so it needs an agreeing `never`.
    assert!(matches!(
        scheme(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": false}"#),
        Err(ConvertError::PrefixSpaceMismatch)
    ));
    assert_eq!(
        scheme(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": false,
                   "prepend_scheme": "never"}"#).unwrap(),
        "never"
    );
    assert!(matches!(
        scheme(r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "sometimes"}"#),
        Err(ConvertError::UnknownPrependScheme { .. })
    ));
    // Checked rather than truncated.
    assert!(matches!(
        decoder(r#"{"type": "Metaspace", "replacement": "ab"}"#),
        Err(ConvertError::MetaspaceBadReplacement { .. })
    ));
    assert!(matches!(
        decoder(r#"{"type": "Metaspace"}"#),
        Err(ConvertError::MetaspaceNoReplacement)
    ));
}

#[test]
fn a_metaspace_pre_tokenizer_becomes_a_normalizer_and_a_split() {
    // Lone: the delimiter half goes to the normalizer slot, the cut stays a `Split`.
    let v = done(BPE, r#", "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}"#);
    assert_eq!(
        v["normalizer"],
        serde_json::json!({"type": "MetaspaceNormalizer", "replacement": "▁",
                           "prepend": true, "drop_whitespace": false})
    );
    assert_eq!(v["pre_tokenizer"]["type"], "Split");
    assert_eq!(v["pre_tokenizer"]["pattern"], serde_json::json!({"String": "▁"}));
    assert_eq!(v["pre_tokenizer"]["behavior"], "MergedWithNext");

    // The t5/albert pair: the `WhitespaceSplit` was never a component, it was `drop_whitespace`.
    let v = done(
        BPE,
        r#", "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
             {"type": "WhitespaceSplit"},
             {"type": "Metaspace", "replacement": "▁"}]}"#,
    );
    assert_eq!(v["normalizer"]["drop_whitespace"], true);
    assert_eq!(v["pre_tokenizer"]["type"], "Split");

    // The delimiter half lands *after* a declared normalizer, which is the order the old reader
    // applied and the one the added-token matcher depends on.
    let v = done(
        BPE,
        r#", "normalizer": {"type": "NFKC"}
           , "pre_tokenizer": {"type": "Metaspace", "replacement": "▁"}"#,
    );
    assert_eq!(v["normalizer"]["normalizers"][0]["type"], "NFKC");
    assert_eq!(v["normalizer"]["normalizers"][1]["type"], "MetaspaceNormalizer");

    // `split: false` wrote delimiters but never cut, and the pair has no way to say that.
    let mut v = config(BPE, r#", "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "split": false}"#);
    assert!(matches!(
        canonicalize_value(&mut v).unwrap_err(),
        ConvertError::MetaspaceNoSplit
    ));
}

// ---- ByteLevel ------------------------------------------------------------------------------

#[test]
fn a_byte_level_pre_tokenizer_becomes_a_model_flag_and_a_split() {
    // `use_regex` defaults to true: the split it asked for is the GPT-2 regex.
    let v = done(BPE, r#", "pre_tokenizer": {"type": "ByteLevel"}"#);
    assert_eq!(v["model"]["byte_level"], true);
    assert_eq!(v["pre_tokenizer"]["pattern"]["Regex"], atomsplit::regexes::GPT2);

    // `use_regex: false` asked only for the byte map, so the member simply goes.
    let v = done(
        BPE,
        r#", "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
             {"type": "WhitespaceSplit"},
             {"type": "ByteLevel", "use_regex": false}]}"#,
    );
    assert_eq!(v["model"]["byte_level"], true);
    assert_eq!(v["pre_tokenizer"]["pretokenizers"][0]["type"], "WhitespaceSplit");
    assert_eq!(v["pre_tokenizer"]["pretokenizers"].as_array().unwrap().len(), 1);

    // A model that never had one still has to say so.
    assert_eq!(done(BPE, "")["model"]["byte_level"], false);

    // Guards the canonical reader no longer carries: it reads `byte_level` only off a BPE, so a
    // flag anywhere else would be silently ignored rather than refused.
    for (extra, want) in [
        (
            r#", "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true}"#,
            "add_prefix_space",
        ),
        (
            r#", "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                 {"type": "ByteLevel"}, {"type": "WhitespaceSplit"}]}"#,
            "last member",
        ),
    ] {
        let mut v = config(BPE, extra);
        let e = canonicalize_value(&mut v).unwrap_err().to_string();
        assert!(e.contains(want), "expected {want:?}, got {e}");
    }
    let mut v = config(
        r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
        r#", "pre_tokenizer": {"type": "ByteLevel"}"#,
    );
    assert!(matches!(
        canonicalize_value(&mut v).unwrap_err(),
        ConvertError::ByteLevelOnNonBpeModel { .. }
    ));
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
    for (text, matches) in [
        ("[]", matches!(canonicalize_str("[]"), Err(ConvertError::NotAnObject { .. }))),
        ("{}", matches!(canonicalize_str("{}"), Err(ConvertError::MissingModel))),
        (r#"{"model": 3}"#, matches!(canonicalize_str(r#"{"model": 3}"#), Err(ConvertError::ModelNotObject { .. }))),
        ("not json", matches!(canonicalize_str("not json"), Err(ConvertError::Json(_)))),
    ] {
        assert!(matches, "{text}");
    }
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
    assert_eq!(once, canonicalize_str(&once).unwrap(), "the pass is not idempotent");
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
                assert!(e.to_string().contains(why), "{name}: not the listed reason: {e}");
                unconvertible += 1;
                continue;
            }
            Ok(c) => {
                assert!(listed.is_none(), "{name}: listed as unconvertible, but it converted");
                c
            }
        };
        assert_no_legacy_residue(&serde_json::from_str(&canonical).unwrap(), &name);
        if let Err(e) = tk_serialize::from_json(&canonical) {
            panic!("{name}: canonicalised, but the canonical reader still refuses it: {e}");
        }
        read += 1;
    }
    eprintln!("{read} converted and read; {unconvertible} with no canonical form; {skipped} not configs");
    assert!(read > 0, "no fixture was actually read");
}
