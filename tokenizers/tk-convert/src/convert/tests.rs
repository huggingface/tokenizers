//! Tests for the canonicalisation pass.
use super::*;

/// Test-data root, spelled the same way `tests/common/mod.rs` spells it.
const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

/// Wrap a `model` (and optionally other top-level fields) in the smallest config that is a
/// valid `tokenizer.json`.
fn config(model: &str, extra: &str) -> Value {
    let text = format!(r#"{{"version": "1.0", "model": {model}{extra}}}"#);
    serde_json::from_str(&text).expect("the test literal is not JSON")
}

fn done(model: &str, extra: &str) -> Value {
    let mut v = config(model, extra);
    canonicalize_value(&mut v).expect("canonicalisation failed");
    assert_no_legacy_residue(&v, "the hand-written literal");
    v
}

fn model_type(v: &Value) -> &str {
    v["model"]["type"].as_str().expect("model has no `type`")
}

/// Assert that not one legacy shape survived the pass.
///
/// This is the check that says what the module is *for*, and it is the one that does not depend
/// on `tk-serialize` still carrying its own backwards-compatibility branches. "The canonical
/// reader accepted the output" is satisfied today by a reader that would also have accepted the
/// input; "the output contains no legacy shape" is the property that lets those branches be
/// deleted, and it is checkable on the JSON alone.
fn assert_no_legacy_residue(v: &Value, what: &str) {
    assert_eq!(
        v.get("version").and_then(Value::as_str),
        Some("2.0"),
        "{what}: the config is still not version 2.0"
    );
    // Both are two components in disguise, and the canonical file spells them as what they are.
    for tag in ["Metaspace", "ByteLevel"] {
        assert!(
            v["pre_tokenizer"].get("type").and_then(Value::as_str) != Some(tag),
            "{what}: the pre_tokenizer slot still holds a `{tag}`"
        );
        assert!(
            !v["pre_tokenizer"]["pretokenizers"]
                .as_array()
                .is_some_and(|m| m
                    .iter()
                    .any(|c| c.get("type").and_then(Value::as_str) == Some(tag))),
            "{what}: a `{tag}` is still a member of the pre_tokenizer `Sequence`"
        );
    }
    let model = v["model"].as_object().expect("no model object");
    if model.get("type").and_then(Value::as_str) == Some("BPE") {
        assert!(
            model.get("byte_level").and_then(Value::as_bool).is_some(),
            "{what}: the BPE model does not say whether it is byte-level"
        );
    }
    assert!(
        model.get("type").and_then(Value::as_str).is_some(),
        "{what}: the model still has no `type`"
    );
    assert!(
        model.get("files").is_none(),
        "{what}: the model still names its vocabulary by path"
    );
    assert!(
        !model.get("vocab").is_some_and(Value::is_string),
        "{what}: `vocab` is still a path"
    );
    if let Some(merges) = model.get("merges").and_then(Value::as_array) {
        for m in merges {
            assert!(
                !m.is_string(),
                "{what}: a merge is still spelled {m} rather than as a pair"
            );
        }
    }
    // Every `Metaspace` anywhere in the tree, not just the ones this pass is known to visit:
    // a residue check that only looked where the walk looks could not catch a missed position.
    fn every_metaspace(node: &Value, what: &str) {
        match node {
            Value::Object(obj) => {
                if obj.get("type").and_then(Value::as_str) == Some("Metaspace") {
                    assert!(
                        obj.get("add_prefix_space").is_none(),
                        "{what}: a Metaspace still spells `add_prefix_space`"
                    );
                    assert!(
                        obj.get("str_rep").is_none(),
                        "{what}: a Metaspace still carries `str_rep`"
                    );
                    assert!(
                        obj.get("prepend_scheme").and_then(Value::as_str).is_some(),
                        "{what}: a Metaspace still has no `prepend_scheme`"
                    );
                    assert!(
                        obj.get("split").is_none(),
                        "{what}: a Metaspace still carries `split`, which only the \
                         pre-tokenizer form had"
                    );
                }
                for v in obj.values() {
                    every_metaspace(v, what);
                }
            }
            Value::Array(items) => {
                for v in items {
                    every_metaspace(v, what);
                }
            }
            _ => {}
        }
    }
    every_metaspace(v, what);
}

// ---------------------------------------------------------------------------------------------
// Rule 1: model type inference
// ---------------------------------------------------------------------------------------------

#[test]
fn fills_a_missing_model_type() {
    // 1. merges ⇒ BPE.
    assert_eq!(
        model_type(&done(
            r#"{"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]}"#,
            ""
        )),
        "BPE"
    );
    // 2. continuing_subword_prefix ⇒ WordPiece.
    assert_eq!(
        model_type(&done(
            r#"{"vocab": {"a": 0}, "continuing_subword_prefix": "@@",
                "unk_token": "[UNK]", "max_input_chars_per_word": 100}"#,
            ""
        )),
        "WordPiece"
    );
    // 3. array-shaped vocab ⇒ Unigram.
    assert_eq!(
        model_type(&done(
            r#"{"vocab": [["a", 0.0], ["b", -1.0]], "unk_id": 0}"#,
            ""
        )),
        "Unigram"
    );
    // 4. nothing else ⇒ WordLevel.
    assert_eq!(
        model_type(&done(r#"{"vocab": {"a": 0}, "unk_token": "<unk>"}"#, "")),
        "WordLevel"
    );
}

/// The ordering trap, and the reason rule 1 is written the way it is.
///
/// A serialized BPE writes every optional field, so it carries `"continuing_subword_prefix":
/// null`. If the WordPiece test ran first, `gpt2.json` -- no `"type"`, string merges, that null
/// prefix -- would come out a WordPiece with no merges, and would either fail to load or
/// tokenize as something else entirely.
#[test]
fn merges_beats_continuing_subword_prefix() {
    let v = done(
        r#"{"dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
            "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}"#,
        "",
    );
    assert_eq!(model_type(&v), "BPE");
    // And the null prefix survives untouched: it is a BPE field too.
    assert!(v["model"]["continuing_subword_prefix"].is_null());
}

/// A non-null `continuing_subword_prefix` with no merges is still a WordPiece, so the ordering
/// above is not just "always BPE".
#[test]
fn a_prefix_without_merges_is_a_wordpiece() {
    let v = done(
        r#"{"vocab": {"[UNK]": 0, "ab": 1, "@@c": 2}, "unk_token": "[UNK]",
            "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100}"#,
        "",
    );
    assert_eq!(model_type(&v), "WordPiece");
}

#[test]
fn an_existing_model_type_is_never_second_guessed() {
    // This *looks* like a BPE by every shape rule, and the tag still wins.
    let v = done(
        r#"{"type": "WordLevel", "vocab": {"a": 0}, "merges": []}"#,
        "",
    );
    assert_eq!(model_type(&v), "WordLevel");
}

// ---------------------------------------------------------------------------------------------
// Legacy merges
// ---------------------------------------------------------------------------------------------

#[test]
fn rewrites_space_joined_merges_into_pairs() {
    let v = done(
        r#"{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b", "ab ab"]}"#,
        "",
    );
    assert_eq!(
        v["model"]["merges"],
        serde_json::json!([["a", "b"], ["ab", "ab"]])
    );
}

#[test]
fn a_merges_array_may_already_be_pairs() {
    let v = done(r#"{"vocab": {"a": 0}, "merges": [["a", "b"], "c d"]}"#, "");
    assert_eq!(
        v["model"]["merges"],
        serde_json::json!([["a", "b"], ["c", "d"]])
    );
}

/// The `merges.txt` header, which a config built by pasting that file's lines still carries. It
/// is dropped rather than converted, because the config path filters it before numbering the
/// ranks -- converting it would push every merge one rank later.
#[test]
fn drops_the_merges_txt_version_header() {
    let v = done(
        r##"{"vocab": {"a": 0}, "merges": ["#version: 0.2", "a b"]}"##,
        "",
    );
    assert_eq!(v["model"]["merges"], serde_json::json!([["a", "b"]]));
}

/// A token containing a space is exactly the ambiguity pairs were introduced to remove. The
/// config path errors (`BadMerges`); guessing here would produce a different tokenizer.
#[test]
fn an_ambiguous_legacy_merge_is_an_error() {
    let mut v = config(r#"{"vocab": {"a": 0}, "merges": ["a b c"]}"#, "");
    let err = canonicalize_value(&mut v).unwrap_err();
    assert!(
        matches!(err, ConvertError::BadMerge { .. }),
        "expected BadMerge, got {err}"
    );
}

// ---------------------------------------------------------------------------------------------
// Unigram's array vocab
// ---------------------------------------------------------------------------------------------

/// The array is left exactly as it is: it is the canonical Unigram shape, and it is also the
/// signal that made rule 3 fire.
#[test]
fn an_array_shaped_unigram_vocab_is_left_alone() {
    let v = done(
        r#"{"vocab": [["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]], "unk_id": 0}"#,
        "",
    );
    assert_eq!(model_type(&v), "Unigram");
    assert_eq!(
        v["model"]["vocab"],
        serde_json::json!([["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]])
    );
}

// ---------------------------------------------------------------------------------------------
// Rule 2: Metaspace
// ---------------------------------------------------------------------------------------------

/// A `Metaspace` in the slot one still lives in. The pre-tokenizer form is two components and is
/// lowered away; the decoder keeps the tag, so it is where the field rules stay observable.
fn metaspace(spelling: &str) -> Result<Value, ConvertError> {
    let mut v = config(
        r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
        &format!(r#", "decoder": {spelling}"#),
    );
    canonicalize_value(&mut v)?;
    Ok(v["decoder"].clone())
}

/// A `Metaspace` pre-tokenizer, lowered. Returns `(normalizer, pre_tokenizer)`.
fn lowered_metaspace(spelling: &str) -> Result<(Value, Value), ConvertError> {
    let mut v = config(
        r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
        &format!(r#", "pre_tokenizer": {spelling}"#),
    );
    canonicalize_value(&mut v)?;
    Ok((v["normalizer"].clone(), v["pre_tokenizer"].clone()))
}

#[test]
fn an_absent_prepend_scheme_is_always_not_never() {
    let ms = metaspace(r#"{"type": "Metaspace", "replacement": "▁"}"#).unwrap();
    assert_eq!(ms["prepend_scheme"], "always");
}

#[test]
fn add_prefix_space_true_is_dropped_and_never_overrides_a_scheme() {
    // Only the old key: agrees with the default.
    let ms =
        metaspace(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#)
            .unwrap();
    assert_eq!(ms["prepend_scheme"], "always");
    assert!(ms.get("add_prefix_space").is_none());

    // Both keys: the explicit scheme wins, and `true` is not treated as a contradiction.
    let ms = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁",
            "add_prefix_space": true, "prepend_scheme": "never"}"#,
    )
    .unwrap();
    assert_eq!(ms["prepend_scheme"], "never");

    let ms = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁",
            "add_prefix_space": true, "prepend_scheme": "first"}"#,
    )
    .unwrap();
    assert_eq!(ms["prepend_scheme"], "first");
}

#[test]
fn add_prefix_space_false_needs_an_agreeing_never() {
    // Agrees: fine, and changes nothing.
    let ms = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁",
            "add_prefix_space": false, "prepend_scheme": "never"}"#,
    )
    .unwrap();
    assert_eq!(ms["prepend_scheme"], "never");

    // Alone: checked against the *defaulted* `always`, so it is a hard error.
    let err =
        metaspace(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": false}"#)
            .unwrap_err();
    assert!(matches!(err, ConvertError::PrefixSpaceMismatch), "{err}");

    // Explicitly disagreeing: also a hard error.
    let err = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁",
            "add_prefix_space": false, "prepend_scheme": "always"}"#,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::PrefixSpaceMismatch), "{err}");

    let err = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁",
            "add_prefix_space": false, "prepend_scheme": "first"}"#,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::PrefixSpaceMismatch), "{err}");
}

#[test]
fn str_rep_is_thrown_away() {
    let ms = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁",
            "str_rep": "▁", "add_prefix_space": true}"#,
    )
    .unwrap();
    assert!(ms.get("str_rep").is_none());
    assert!(ms.get("add_prefix_space").is_none());
}

/// `split: false` asked the `Metaspace` to rewrite the text without cutting it, and the canonical
/// pair has no way to say that -- the `Split` is the cut.
#[test]
fn a_metaspace_that_does_not_split_has_no_canonical_form() {
    let err = lowered_metaspace(
        r#"{"type": "Metaspace", "replacement": "▁", "split": false}"#,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::MetaspaceNoSplit), "{err}");
}

#[test]
fn an_unknown_prepend_scheme_is_refused() {
    let err = metaspace(
        r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "sometimes"}"#,
    )
    .unwrap_err();
    assert!(
        matches!(err, ConvertError::UnknownPrependScheme { .. }),
        "{err}"
    );
}

#[test]
fn a_metaspace_replacement_must_be_one_character() {
    let err = metaspace(r#"{"type": "Metaspace"}"#).unwrap_err();
    assert!(matches!(err, ConvertError::MetaspaceNoReplacement), "{err}");
    let err = metaspace(r#"{"type": "Metaspace", "replacement": "__"}"#).unwrap_err();
    assert!(
        matches!(err, ConvertError::MetaspaceBadReplacement { .. }),
        "{err}"
    );
}

/// t5 and albert both spell `Metaspace` twice -- inside a pre-tokenizer `Sequence` and again as
/// the decoder -- and both copies carry the legacy spelling. All three positions plus the
/// nesting are exercised here in one config.
#[test]
fn every_metaspace_position_is_walked() {
    let mut v = config(
        r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
        r#", "normalizer": {"type": "Sequence", "normalizers": [
               {"type": "Metaspace", "replacement": "▁", "str_rep": "▁"}]}
           , "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
               {"type": "WhitespaceSplit"},
               {"type": "Sequence", "pretokenizers": [
                 {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}]}]}
           , "decoder": {"type": "Sequence", "decoders": [
               {"type": "Metaspace", "replacement": "▁", "str_rep": "▁",
                "add_prefix_space": true}]}"#,
    );
    canonicalize_value(&mut v).unwrap();

    for found in [
        &v["normalizer"]["normalizers"][0],
        &v["decoder"]["decoders"][0],
    ] {
        assert_eq!(found["prepend_scheme"], "always");
        assert!(found.get("split").is_none());
        assert!(found.get("add_prefix_space").is_none());
        assert!(found.get("str_rep").is_none());
    }
}

/// A `Metaspace` inside a post-processor's `special_tokens` is *data*, not a component, and must
/// not be rewritten. This pins the "only descend through the four `Sequence` child keys"
/// decision.
#[test]
fn data_objects_are_not_mistaken_for_components() {
    let mut v = config(
        r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
        r#", "post_processor": {"type": "TemplateProcessing", "single": [],
              "pair": [], "special_tokens": {"x": {"type": "Metaspace"}}}"#,
    );
    // No `replacement`, so if the component walk reached it this would be
    // `MetaspaceNoReplacement`. It does not: the table is data, and the template lowering consumes
    // it rather than descending into it.
    canonicalize_value(&mut v).unwrap();
    assert!(v["post_processor"].get("special_tokens").is_none());
}

// ---------------------------------------------------------------------------------------------
// Shape of the pass itself
// ---------------------------------------------------------------------------------------------

#[test]
fn refuses_something_that_is_not_a_tokenizer_config() {
    assert!(matches!(
        canonicalize_str("[]").unwrap_err(),
        ConvertError::NotAnObject { .. }
    ));
    assert!(matches!(
        canonicalize_str("{}").unwrap_err(),
        ConvertError::MissingModel
    ));
    assert!(matches!(
        canonicalize_str(r#"{"model": 3}"#).unwrap_err(),
        ConvertError::ModelNotObject { .. }
    ));
    assert!(matches!(
        canonicalize_str("not json").unwrap_err(),
        ConvertError::Json(_)
    ));
}

/// Canonicalising an already-canonical file must be a no-op that still succeeds -- that is what
/// makes it safe to run unconditionally in front of a reader.
#[test]
fn is_idempotent() {
    let legacy = r##"{"version": "1.0",
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": "▁", "str_rep": "▁",
             "add_prefix_space": true}]},
        "decoder": {"type": "Metaspace", "replacement": "▁", "str_rep": "▁",
                    "add_prefix_space": true},
        "model": {"continuing_subword_prefix": null, "vocab": {"a": 0, "b": 1, "ab": 2},
                  "merges": ["#version: 0.2", "a b"]}}"##;

    let once = canonicalize_str(legacy).unwrap();
    let twice = canonicalize_str(&once).unwrap();
    assert_eq!(once, twice, "the pass is not idempotent");
    assert_no_legacy_residue(&serde_json::from_str(&once).unwrap(), "the legacy literal");

    // And the third pass over the *value* API agrees with the string one.
    let mut v: Value = serde_json::from_str(&twice).unwrap();
    canonicalize_value(&mut v).unwrap();
    assert_eq!(serde_json::to_string_pretty(&v).unwrap(), twice);
}

// ---------------------------------------------------------------------------------------------
// End to end: every fixture, through the canonical reader
// ---------------------------------------------------------------------------------------------

/// The only test that proves the point of the module: after this pass, the *canonical* reader
/// can read the file. `tk_serialize::from_json` is a dev-dependency with every component
/// feature on, precisely so this cannot silently drop to "read nothing and pass".
///
/// ## Why the reader's verdict is compared before *and* after
///
/// "The canonical reader accepts the output" is not quite the property to assert, because that
/// reader also refuses things for reasons that have nothing to do with the file's age: the
/// pipeline cannot express a `ByteLevel` with `add_prefix_space: true`, or a `Metaspace` with
/// `prepend_scheme: first`, and no amount of converting will change that. `data/tokenizer.json`
/// is exactly that case, and asserting "everything reads" would leave this test with a
/// hand-maintained exception list that quietly grows.
///
/// So the assertion is the one a converter actually owes its caller, in two halves:
///
/// - **no regression** — a file the reader accepted before must still be accepted after;
/// - **no residue** — a file the reader still refuses must fail with the *same error it already
///   failed with*. An unchanged message is proof the refusal is about a component the pipeline
///   cannot build, not about a field this pass was supposed to fill.
///
/// The err → ok column is the conversion doing visible work. It is small today only because
/// `tk-serialize` has not yet dropped its own three legacy branches; when it does, this column
/// is what will carry the fixtures that only load through this pass.
///
/// Two fixtures in `data/` are not tokenizer configs and are reported as skipped rather than
/// failed: `gpt2-vocab.json` is a bare `{token: id}` map (which happens to contain a token
/// spelled `model`, hence the "must be an object" check rather than a "has a model" one) and
/// `unigram.json` is a bare Unigram model with no surrounding config.
///
/// A missing `data/` skips the whole test: the directory is gitignored and populated by
/// `make fixtures`, so a fresh checkout has none of it.
#[test]
fn every_fixture_canonicalises_into_something_the_canonical_reader_accepts() {
    /// Fixtures with no canonical form, each with a substring its refusal must contain.
    ///
    /// Every entry is a limit of what the pipeline can *build*, not a gap in this pass. Listing
    /// them here rather than skipping makes adding one a diff a reviewer sees.
    const UNCONVERTIBLE: &[(&str, &str)] = &[
        // A `ByteLevel` with `add_prefix_space: true`, which the pipeline cannot express.
        ("tokenizer.json", "add_prefix_space"),
    ];

    let dir = std::path::Path::new(DATA);
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("skipping: no fixture directory at {DATA}");
        return;
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("skipping: no *.json fixtures in {DATA}");
        return;
    }

    let read = |text: &str| {
        tk_serialize::from_json(text)
            .map(|_| ())
            .map_err(|e| e.to_string())
    };

    let (mut ok, mut skipped, mut unconvertible) = (0usize, 0usize, 0usize);
    for (i, path) in files.iter().enumerate() {
        let at = format!("[{}/{}]", i + 1, files.len());
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(path).unwrap();
        let parsed: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => panic!("{at} {name}: not JSON at all: {e}"),
        };
        if !parsed.get("model").is_some_and(Value::is_object) {
            eprintln!("{at} {name}  skipped (not a tokenizer.json)");
            skipped += 1;
            continue;
        }

        // Refusing is a legitimate outcome, but only for a listed file: some legacy spellings
        // have no canonical form and inventing one would move ids. Both directions are checked --
        // a stale entry fails, and a refusal for an unlisted reason fails.
        let listed = UNCONVERTIBLE.iter().find(|(n, _)| *n == name);
        let canonical = match canonicalize_file(path) {
            Err(e) => {
                let (_, why) = listed.unwrap_or_else(|| {
                    panic!("{at} {name}: this pass refuses to convert it: {e}")
                });
                assert!(
                    e.to_string().contains(why),
                    "{at} {name}: refused, but not for the listed reason ({why}): {e}"
                );
                eprintln!("{at} {name}  unconvertible, as listed: {e}");
                unconvertible += 1;
                continue;
            }
            Ok(c) => {
                assert!(
                    listed.is_none(),
                    "{at} {name}: listed as unconvertible, but it converted -- drop the entry"
                );
                c
            }
        };
        // The check that does not depend on what the reader currently tolerates.
        assert_no_legacy_residue(&serde_json::from_str(&canonical).unwrap(), &name);
        // Absolute, not "no worse than before": every fixture either converts and reads, or is
        // named in `UNCONVERTIBLE`. The old before/after comparison cannot say anything any more --
        // the reader refuses every raw 1.0 file, so "before" is the same version error every time.
        match read(&canonical) {
            Ok(()) => {
                eprintln!("{at} {name}  ok");
                ok += 1;
            }
            Err(now) => panic!(
                "{at} {name}: canonicalised, but the canonical reader still refuses it: {now}"
            ),
        }
    }
    eprintln!(
        "{ok} converted and read through the canonical reader; {unconvertible} with no canonical \
         form; {skipped} not tokenizer configs"
    );
    assert!(ok > 0, "no fixture was actually read");
}
