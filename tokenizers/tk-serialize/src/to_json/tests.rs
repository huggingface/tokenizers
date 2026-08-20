//! The writer's tests. The one that matters is [`round_trip_preserves_ids_on_every_real_config`].
//!
//! The contract this writer offers is *ids*, not bytes — the pipeline is a lowered form and cannot
//! reproduce the file it came from (see the module docs). So the gate is: read a real config, write
//! it, read that back, and encode with both. Anything the writer gets wrong about a component shows
//! up as a different id, which is the only thing a caller can observe.
//!
//! Everything else here supports that one. The float sweep pins the single value most likely to
//! shift silently and least likely to show up in a fixed corpus; the canonical-spelling tests pin
//! the shapes rather than the behaviour; and the per-component tests localise a failure the
//! round-trip would only report as "ids moved".

use super::*;
use crate::from_json::{from_json, from_json_file};

/// A minimal non-byte-level BPE: no data files, no regex backend, two merges over four tokens.
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
#[cfg(any(feature = "unigram", feature = "wordpiece"))]
const TINY_BPE_MODEL: &str = r#"{
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
        "merges": [["a", "b"], ["ab", "ab"]]
    }"#;

/// [`TINY_BPE`] with one top-level component filled in. Every such field is `null` there, so a plain
/// textual replace is unambiguous.
fn with_component(field: &str, json: &str) -> String {
    TINY_BPE.replace(
        &format!(r#""{field}": null"#),
        &format!(r#""{field}": {json}"#),
    )
}

/// [`TINY_BPE`] with its model swapped out.
#[cfg(any(feature = "unigram", feature = "wordpiece"))]
fn with_model(model: &str) -> String {
    TINY_BPE.replace(TINY_BPE_MODEL, model)
}

/// Read a config, write it back, and hand over the text.
fn rewrite(text: &str) -> String {
    let tokenizer = from_json(text).expect("the config reads");
    to_json(&tokenizer).expect("a config that reads should write")
}

/// One top-level field of a written config, as `serde_json` sees it.
///
/// `serde_json` is a dev-dependency, used here as an *independent* parser: it is what makes "the
/// writer emits valid JSON" a real assertion rather than a check of our own parser against itself.
fn field_of(written: &str, field: &str) -> serde_json::Value {
    let parsed: serde_json::Value =
        serde_json::from_str(written).expect("the writer emits valid JSON");
    parsed
        .get(field)
        .unwrap_or_else(|| panic!("the written config has no `{field}`"))
        .clone()
}

/// The component a config round-trips to, so a test can state what it expects in JSON.
fn component_round_trip(field: &str, json: &str) -> serde_json::Value {
    field_of(&rewrite(&with_component(field, json)), field)
}

fn ids(tokenizer: &PipelineTokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
    tokenizer
        .encode(text, add_special_tokens)
        .wait()
        .expect("encoding a fixture text")
        .iter()
        .flat_map(|encoding| encoding.ids())
        .map(|token| token.id())
        .collect()
}

/// The fixture configs, read from the directory rather than hard-coded so a new `make models` file
/// is covered the day it lands.
///
/// `data/tokenizer.json` is excluded by name: it is not a fixture but test *output*, written by
/// `tests/documentation.rs` with `.save(..)`. Cargo runs test binaries in parallel, so reading it
/// here races that write.
fn fixtures() -> Vec<std::path::PathBuf> {
    let dir = std::path::Path::new("../data");
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read data/")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name != "tokenizer.json")
        })
        .collect();
    files.sort();
    files
}

/// One text per encode regime rather than one per language: latin words, code, CJK without spaces,
/// an RTL abjad with combining marks, Cyrillic, digits, whitespace-only, empty, and — the one that
/// catches an added-token replay in the wrong order — text sprinkled with the markers real chat
/// traffic carries.
const TEXTS: &[&str] = &[
    " the quick brown fox jumps over the lazy dog",
    "def foo(bar):\n    return bar + 1\n",
    " 语言模型 mixed with ASCII and ελληνικά",
    "الْعَرَبِيَّة и русский текст",
    "unprefixed internationalisation",
    "3.14159 and 1e-9 and 0007",
    "[CLS] <s> <|endoftext|> [SEP] </s> <unk> [MASK]",
    "tabs\tand\nnewlines\r\nand  double  spaces",
    "   ",
    "",
];

/// Every `data/*.json`, read then written then read, compared id for id.
///
/// This is the gate. A config the *reader* refuses is skipped and named on stderr, because the
/// refusals are real: `gpt2-vocab.json` is a bare vocabulary rather than a tokenizer, `unigram.json`
/// has no model, and a `Split` on a regex the native FSM does not recognise needs a backend this
/// crate's test build does not enable.
///
/// A config the reader accepts and the *writer* refuses is a failure, not a skip: everything the
/// reader can build, the writer is supposed to be able to describe.
///
/// The `>= 1` guard is the only count asserted. Anything stronger would encode which fixtures a
/// given machine has downloaded, and CI's set (`TESTS_RESOURCES`, eight configs, no `t5-base.json`)
/// is fourteen short of a full `make models`.
#[test]
fn round_trip_preserves_ids_on_every_real_config() {
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip all: ../data is missing or empty -- `make data models` fetches fixtures");
        return;
    }

    let (mut configs, mut comparisons) = (0usize, 0usize);
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read a fixture");
        let before = match from_json(&text) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                eprintln!("  skip {name}: the reader refused it: {e}");
                continue;
            }
        };
        let written = match to_json(&before) {
            Ok(written) => written,
            Err(e) => panic!("{name}: the reader built it but the writer cannot describe it: {e}"),
        };
        let after = match from_json(&written) {
            Ok(tokenizer) => tokenizer,
            Err(e) => panic!("{name}: the writer produced a config its own reader refuses: {e}"),
        };

        configs += 1;
        for text in TEXTS {
            for add_special_tokens in [false, true] {
                assert_eq!(
                    ids(&before, text, add_special_tokens),
                    ids(&after, text, add_special_tokens),
                    "{name}: ids moved across a write/read round trip \
                     (add_special_tokens={add_special_tokens}) on {text:?}"
                );
                comparisons += 1;
            }
        }
        eprintln!("  ok {name}: {} comparisons", TEXTS.len() * 2);
    }
    eprintln!("round trip: {configs} configs, {comparisons} comparisons");
    assert!(
        configs >= 1,
        "no fixture round-tripped; every config in ../data was skipped"
    );
}

/// The float rule, over every Unigram score in every Unigram config in `data/`.
///
/// Not a sample. A Unigram score feeds a Viterbi lattice, so a one-ULP shift in one of 250,002 of
/// them can flip a near-tie and move an id — and it would not show up above unless the tie happened
/// to fall inside `TEXTS`. So the invariant is checked directly, on every value: the literal the
/// writer emits must read back, *through this crate's parser*, to the same bits.
///
/// The parser is the point. `json.rs` reproduces `serde_json`'s default arithmetic rather than being
/// correctly rounded, deliberately, so "the shortest form that round-trips" — a property defined
/// against a correctly-rounded parser — is not automatically true here. It is measured, not assumed.
///
/// Needs `normalizers` as well as `unigram`, and that is not belt-and-braces: every Unigram config
/// in `data/` -- t5-base, albert, albert-base-v1, xlmr -- carries a SentencePiece normalizer chain
/// (`Precompiled`, `NFKD`, `StripAccents`), so with that feature off the reader refuses all four and
/// there is nothing to sweep. Asserting `>= 1` in that build would be asserting a feature set rather
/// than a fixture set.
#[test]
#[cfg(all(feature = "unigram", feature = "normalizers"))]
fn every_unigram_score_survives_the_writer_bit_for_bit() {
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip all: ../data is missing or empty -- `make data models` fetches fixtures");
        return;
    }

    let (mut models, mut scores, mut widest) = (0usize, 0usize, 0usize);
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read a fixture");
        let tokenizer = match from_json(&text) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                eprintln!("  skip {name}: the reader refused it: {e}");
                continue;
            }
        };
        // The model's kind from the *built pipeline*, never from the raw file. Grepping the text for
        // `"Unigram"` would miss `albert-base-v1-tokenizer.json`, whose model is a bare
        // `{unk_id, vocab}` with no `"type"` at all.
        let PipelineModel::Unigram(unigram) = tokenizer.get_model() else {
            continue;
        };
        models += 1;
        let mut file_widest = 0usize;
        for (token, score) in unigram.vocab() {
            let literal = super::writer::float_literal(*score)
                .unwrap_or_else(|e| panic!("{name}: the score for {token:?} has no spelling: {e}"));
            let read_back = crate::json::f64_from_literal(&literal);
            assert_eq!(
                read_back.to_bits(),
                score.to_bits(),
                "{name}: the score for {token:?} was written as {literal}, which this crate's \
                 parser reads back as {read_back} ({:016x}) instead of {score} ({:016x})",
                read_back.to_bits(),
                score.to_bits()
            );
            // Significant digits, which is the number worth knowing: it says how close the emitted
            // form sits to the 17 that pin an `f64`.
            let digits = literal
                .split(['e', 'E'])
                .next()
                .unwrap_or_default()
                .chars()
                .filter(|c| c.is_ascii_digit())
                .count();
            file_widest = file_widest.max(digits);
            scores += 1;
        }
        widest = widest.max(file_widest);
        eprintln!(
            "  ok {name}: {} scores, at most {file_widest} significant digits",
            unigram.vocab().len()
        );
    }
    eprintln!("float sweep: {models} Unigram models, {scores} scores, at most {widest} digits");
    assert!(
        models >= 1,
        "no Unigram config was checked; `make data` fetches albert-base-v1-tokenizer.json and \
         `make models` fetches t5-base.json"
    );
}

/// A BPE whose two merges *compete* for the same symbol: `a+b` and `b+c` both apply to `abc`, and
/// only their ranks decide which wins.
///
/// [`TINY_BPE`] cannot serve here, and the reason is worth writing down. Its merges form a *chain* —
/// `a+b` makes `ab`, then `ab+ab` makes `abab` — and a chain is immune to reordering: BPE applies
/// whichever pair is present, and at each step only one is, so `abab` comes out as a single token
/// whatever the ranks say. Reversing a chain is as inert as `merges.pop()`.
const COMPETING_BPE: &str = r#"{
    "version": "1.0",
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4},
        "merges": [["a", "b"], ["b", "c"]]
    }
}"#;

/// The perturbation that proves the round-trip gate has teeth.
///
/// Finding one that actually bites took two attempts, and both failures are the same mistake:
///
/// - `merges.pop()` does nothing. The lowest-priority pair is the one least likely to be reached at
///   all, and on a byte-level model the byte-level fold covers it anyway.
/// - `merges.reverse()` does nothing *either*, as long as the merges form a chain — see
///   [`COMPETING_BPE`]. A reversed chain still produces the same tokens, because BPE applies
///   whatever pair is in front of it and a chain never offers a choice.
///
/// What bites is reversing a list whose merges **compete**: two pairs applicable to the same
/// position, where the ranks are the only tie-break. Then reversing swaps the winner.
///
/// Applied to the *written config* rather than to the writer, so the test cannot rot when the writer
/// changes: what it demonstrates is that the comparison
/// [`round_trip_preserves_ids_on_every_real_config`] performs is sensitive to a wrong merge order,
/// whoever produced it.
#[test]
fn reversing_the_written_merges_moves_ids() {
    let tokenizer = from_json(COMPETING_BPE).expect("the competing config reads");
    let written = to_json(&tokenizer).expect("and writes");

    let mut parsed: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");
    let merges = parsed["model"]["merges"]
        .as_array_mut()
        .expect("a BPE model has a merges array");
    assert_eq!(
        *merges,
        vec![serde_json::json!(["a", "b"]), serde_json::json!(["b", "c"])],
        "the writer did not reproduce the merge list in rank order"
    );
    merges.reverse();
    let perturbed = from_json(&parsed.to_string()).expect("the perturbed config still reads");

    // `a+b` outranks `b+c`, so `abc` becomes `ab` + `c`. Reversed, `b+c` wins and it becomes
    // `a` + `bc` — the same three characters, different ids.
    assert_eq!(
        ids(&tokenizer, "abc", false),
        vec![3, 2],
        "the unperturbed model should prefer the higher-ranked `a`+`b`"
    );
    assert_eq!(
        ids(&perturbed, "abc", false),
        vec![0, 4],
        "reversing the merge order left the ids unchanged, so the round-trip gate is not \
         actually comparing anything"
    );
}

/// The two weaker perturbations, recorded as tests so nobody has to rediscover that they are inert.
///
/// Both leave the ids alone, and both look like a working gate has died. Keeping them here says out
/// loud that a passing round-trip is not evidence the perturbation was too weak to matter.
#[test]
fn the_weak_perturbations_really_are_inert() {
    let tokenizer = from_json(TINY_BPE).expect("the tiny config reads");
    let written = to_json(&tokenizer).expect("and writes");
    let baseline = ids(&tokenizer, "abab", false);
    assert_eq!(baseline, vec![3], "`abab` merges all the way up");

    for (what, perturb) in [
        ("popping the last merge", true),
        ("reversing a chain of merges", false),
    ] {
        let mut parsed: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");
        let merges = parsed["model"]["merges"].as_array_mut().expect("merges");
        if perturb {
            merges.pop();
        } else {
            merges.reverse();
        }
        let Ok(perturbed) = from_json(&parsed.to_string()) else {
            // Being refused outright is a *detection*, which would make the perturbation a fine one.
            panic!("{what} was refused, so it is not the inert case this test documents");
        };
        assert_eq!(
            ids(&perturbed, "abab", false),
            baseline,
            "{what} moved an id after all -- if that is now true, this test should become a real \
             perturbation rather than a warning about a weak one"
        );
    }
}

/// The same perturbation against the real fixtures, so the gate is shown to bite on the configs it
/// actually guards rather than only on a four-token toy.
///
/// Skipped per config when reversing changes nothing observable, which is a legitimate outcome for a
/// vocabulary where `TEXTS` happens to reach no multi-merge token — but not for all of them at once,
/// hence the `>= 1`.
#[test]
fn reversing_the_written_merges_moves_ids_on_a_real_config() {
    let files = fixtures();
    if files.is_empty() {
        eprintln!("skip all: ../data is missing or empty -- `make data models` fetches fixtures");
        return;
    }
    let mut bitten = 0usize;
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read a fixture");
        let Ok(before) = from_json(&text) else {
            continue;
        };
        let Ok(written) = to_json(&before) else {
            continue;
        };
        let mut parsed: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");
        let Some(merges) = parsed["model"]["merges"].as_array_mut() else {
            continue; // not a BPE
        };
        if merges.len() < 2 {
            continue;
        }
        merges.reverse();
        // A reversed merge list can be *rejected* rather than merely different — a byte-level model
        // demands every byte be an atom, and inverting the ranks can break that. Either way the
        // perturbation was detected, which is what this test is about.
        let Ok(after) = from_json(&parsed.to_string()) else {
            eprintln!("  {name}: reversed merges are refused outright");
            bitten += 1;
            continue;
        };
        let moved = TEXTS
            .iter()
            .any(|text| ids(&before, text, false) != ids(&after, text, false));
        if moved {
            eprintln!("  {name}: reversed merges move ids");
            bitten += 1;
        } else {
            eprintln!("  skip {name}: reversing its merges changed no id in TEXTS");
        }
    }
    assert!(
        bitten >= 1,
        "reversing the merge list of every BPE fixture changed nothing, which would mean the \
         round-trip comparison cannot see a merge-order defect at all"
    );
}

// ---- canonical spelling ----------------------------------------------------------------------

/// Every component carries an explicit `"type"`. This is what makes the output canonical rather than
/// merely readable: the reader still tolerates an untagged model, and the writer must never make one.
#[test]
fn every_component_is_written_with_a_type_tag() {
    let config = with_component("normalizer", r#"{"type": "Lowercase"}"#);
    let config = with_component_in(&config, "decoder", r#"{"type": "Fuse"}"#);
    let config = with_component_in(&config, "pre_tokenizer", r#"{"type": "Whitespace"}"#);
    let parsed: serde_json::Value = serde_json::from_str(&rewrite(&config)).expect("valid JSON");
    for field in ["normalizer", "pre_tokenizer", "decoder", "model"] {
        assert!(
            parsed[field]["type"].as_str().is_some(),
            "`{field}` was written without a `type`: {}",
            parsed[field]
        );
    }
    assert_eq!(parsed["model"]["type"], "BPE");
    assert_eq!(parsed["version"], "1.0");
}

/// `with_component`, but against an already-modified config.
fn with_component_in(config: &str, field: &str, json: &str) -> String {
    config.replace(
        &format!(r#""{field}": null"#),
        &format!(r#""{field}": {json}"#),
    )
}

/// Merges go out as `["a", "b"]` pairs, never as the legacy `"a b"` string — which is ambiguous
/// exactly when a token contains a space, and is why the pair form became canonical.
#[test]
fn merges_are_written_as_pairs_in_rank_order() {
    let merges = field_of(&rewrite(TINY_BPE), "model")["merges"].clone();
    assert_eq!(
        merges,
        serde_json::json!([["a", "b"], ["ab", "ab"]]),
        "merges are not canonical pairs in rank order"
    );
}

/// A legacy config in, a canonical one out. `gpt2`-style files ship both of these, and the point of
/// the writer is that reading one and writing it produces the modern spelling.
#[test]
fn a_legacy_config_is_written_canonically() {
    let legacy = r#"{
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
            "merges": ["a b", "ab ab"]
        }
    }"#;
    let model = field_of(&rewrite(legacy), "model");
    // The model had no `"type"` and its merges were strings. Both are canonical now.
    assert_eq!(model["type"], "BPE");
    assert_eq!(
        model["merges"],
        serde_json::json!([["a", "b"], ["ab", "ab"]])
    );
}

// ---- components, one test each ----------------------------------------------------------------
//
// The round-trip test catches any of these as "ids moved", without saying where. These say where.

#[test]
fn a_normalizer_sequence_is_written_flat() {
    // Nested on the way in; the pipeline holds only the flattening, so that is what comes out. Both
    // read back as the same chain, which is the property that matters.
    let written = component_round_trip(
        "normalizer",
        r#"{"type": "Sequence", "normalizers": [
            {"type": "Lowercase"},
            {"type": "Sequence", "normalizers": [
                {"type": "Strip", "strip_left": true, "strip_right": false}
            ]},
            {"type": "Prepend", "prepend": "_"}
        ]}"#,
    );
    assert_eq!(
        written,
        serde_json::json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Lowercase"},
                {"type": "Strip", "strip_left": true, "strip_right": false},
                {"type": "Prepend", "prepend": "_"}
            ]
        })
    );
}

/// A single normalizer goes out on its own, not wrapped in a one-member `Sequence`.
#[test]
fn a_lone_normalizer_is_not_wrapped() {
    assert_eq!(
        component_round_trip("normalizer", r#"{"type": "Lowercase"}"#),
        serde_json::json!({"type": "Lowercase"}),
    );
}

#[cfg(feature = "normalizers")]
#[test]
fn a_bert_normalizer_keeps_its_null_strip_accents() {
    // `null` is not "absent": it means "decide from `lowercase`", and the reader requires the key to
    // be there for exactly that reason.
    assert_eq!(
        component_round_trip(
            "normalizer",
            r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                "strip_accents": null, "lowercase": true}"#
        ),
        serde_json::json!({
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": true,
            "strip_accents": null,
            "lowercase": true
        })
    );
}

#[test]
fn a_replace_keeps_its_externally_tagged_pattern() {
    assert_eq!(
        component_round_trip(
            "normalizer",
            r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#
        ),
        serde_json::json!({"type": "Replace", "pattern": {"String": " "}, "content": "_"})
    );
}

#[test]
fn absent_components_are_written_as_null() {
    let written = rewrite(TINY_BPE);
    for field in ["normalizer", "pre_tokenizer", "decoder"] {
        assert_eq!(
            field_of(&written, field),
            serde_json::Value::Null,
            "`{field}` should be absent"
        );
    }
    assert_eq!(
        field_of(&written, "post_processor"),
        serde_json::Value::Null,
        "a pass-through frame is what `no post-processor` lowers to, so it goes back out as absent"
    );
    // Neither is part of a pipeline, and both have always been spelled out.
    assert_eq!(field_of(&written, "truncation"), serde_json::Value::Null);
    assert_eq!(field_of(&written, "padding"), serde_json::Value::Null);
}

/// An empty `Sequence` normalizer disappears on the way in — deepseek ships one — so it comes back
/// as no normalizer rather than as an empty `Sequence`.
#[test]
fn an_empty_normalizer_sequence_disappears() {
    assert_eq!(
        component_round_trip("normalizer", r#"{"type": "Sequence", "normalizers": []}"#),
        serde_json::Value::Null
    );
}

#[test]
fn pre_tokenizers_keep_their_fields() {
    for (input, expected) in [
        (
            r#"{"type": "Digits", "individual_digits": true}"#,
            serde_json::json!({"type": "Digits", "individual_digits": true}),
        ),
        (
            r#"{"type": "Whitespace"}"#,
            serde_json::json!({"type": "Whitespace"}),
        ),
        (
            r#"{"type": "WhitespaceSplit"}"#,
            serde_json::json!({"type": "WhitespaceSplit"}),
        ),
        (
            r#"{"type": "BertPreTokenizer"}"#,
            serde_json::json!({"type": "BertPreTokenizer"}),
        ),
        (
            r#"{"type": "Punctuation", "behavior": "Removed"}"#,
            serde_json::json!({"type": "Punctuation", "behavior": "Removed"}),
        ),
        (
            r#"{"type": "CharDelimiterSplit", "delimiter": "-"}"#,
            serde_json::json!({"type": "CharDelimiterSplit", "delimiter": "-"}),
        ),
        (
            r#"{"type": "FixedLength", "length": 7}"#,
            serde_json::json!({"type": "FixedLength", "length": 7}),
        ),
        (
            r#"{"type": "Split", "pattern": {"String": "-"},
                "behavior": "MergedWithNext", "invert": false}"#,
            serde_json::json!({
                "type": "Split",
                "pattern": {"String": "-"},
                "behavior": "MergedWithNext",
                "invert": false
            }),
        ),
        (
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "Whitespace"}, {"type": "Digits", "individual_digits": false}
            ]}"#,
            serde_json::json!({
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Whitespace"},
                    {"type": "Digits", "individual_digits": false}
                ]
            }),
        ),
    ] {
        assert_eq!(
            component_round_trip("pre_tokenizer", input),
            expected,
            "{input}"
        );
    }
}

/// A `Metaspace` pre-tokenizer went in as a normalizer plus a `Split`; it has to come back out as a
/// `Metaspace`, spelled the canonical way. This is the inversion most likely to break.
#[test]
fn a_metaspace_pre_tokenizer_is_rebuilt_from_its_normalizer() {
    let written = component_round_trip(
        "pre_tokenizer",
        r#"{"type": "Metaspace", "replacement": "▁",
            "prepend_scheme": "always", "split": true}"#,
    );
    assert_eq!(
        written,
        serde_json::json!({
            "type": "Metaspace",
            "replacement": "\u{2581}",
            "prepend_scheme": "always",
            "split": true
        })
    );
    // And the normalizer it lowered into is not *also* written, which would apply the rewrite twice.
    let config = with_component(
        "pre_tokenizer",
        r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}"#,
    );
    assert_eq!(
        field_of(&rewrite(&config), "normalizer"),
        serde_json::Value::Null,
        "the Metaspace normalizer was written as a `normalizer` as well as a pre-tokenizer"
    );
}

/// `add_prefix_space` on the way in, `prepend_scheme` on the way out: the legacy key is never
/// written, which is what "canonical" means here.
#[test]
fn a_metaspace_never_goes_out_with_add_prefix_space() {
    let written = component_round_trip(
        "pre_tokenizer",
        r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
    );
    assert_eq!(written["prepend_scheme"], "always");
    assert!(
        written.get("add_prefix_space").is_none(),
        "the legacy key was written back: {written}"
    );
}

/// t5 and albert's `Sequence[WhitespaceSplit, Metaspace]`, which the reader collapses into a single
/// `Metaspace` normalizer carrying `drop_whitespace`. The pair has to come back, because a lone
/// `Metaspace` is a different pipeline.
#[test]
fn a_whitespace_split_metaspace_pair_comes_back_as_a_pair() {
    assert_eq!(
        component_round_trip(
            "pre_tokenizer",
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {"type": "Metaspace", "replacement": "▁",
                 "prepend_scheme": "always", "split": true}
            ]}"#
        ),
        serde_json::json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {
                    "type": "Metaspace",
                    "replacement": "\u{2581}",
                    "prepend_scheme": "always",
                    "split": true
                }
            ]
        })
    );
}

#[test]
fn decoders_keep_their_fields() {
    for (input, expected) in [
        (r#"{"type": "Fuse"}"#, serde_json::json!({"type": "Fuse"})),
        (
            r#"{"type": "ByteFallback"}"#,
            serde_json::json!({"type": "ByteFallback"}),
        ),
        (
            r#"{"type": "Strip", "content": "_", "start": 1, "stop": 0}"#,
            serde_json::json!({"type": "Strip", "content": "_", "start": 1, "stop": 0}),
        ),
        (
            r#"{"type": "BPEDecoder", "suffix": "</w>"}"#,
            serde_json::json!({"type": "BPEDecoder", "suffix": "</w>"}),
        ),
        (
            r###"{"type": "WordPiece", "prefix": "##", "cleanup": true}"###,
            serde_json::json!({"type": "WordPiece", "prefix": "##", "cleanup": true}),
        ),
        (
            r#"{"type": "CTC", "pad_token": "<pad>",
                "word_delimiter_token": "|", "cleanup": true}"#,
            serde_json::json!({
                "type": "CTC",
                "pad_token": "<pad>",
                "word_delimiter_token": "|",
                "cleanup": true
            }),
        ),
        (
            r#"{"type": "ByteLevel", "add_prefix_space": true,
                "trim_offsets": false, "use_regex": false}"#,
            serde_json::json!({
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false,
                "use_regex": false
            }),
        ),
        (
            // A decoder keeps all three prepend schemes: nothing here has to be expressible as a
            // normalizer, so `first` survives where the pre-tokenizer's does not.
            r#"{"type": "Metaspace", "replacement": "▁",
                "prepend_scheme": "first", "split": false}"#,
            serde_json::json!({
                "type": "Metaspace",
                "replacement": "\u{2581}",
                "prepend_scheme": "first",
                "split": false
            }),
        ),
        (
            r#"{"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}"#,
            serde_json::json!({
                "type": "Sequence",
                "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]
            }),
        ),
    ] {
        assert_eq!(component_round_trip("decoder", input), expected, "{input}");
    }
}

/// Every post-processor becomes a `TemplateProcessing`, because that is what the pipeline holds: two
/// templates of placeholders and resolved ids. The names are rebuilt from the ids, so this checks
/// that the rebuilt table resolves to the same frame.
#[test]
fn a_post_processor_is_written_as_a_template() {
    assert_eq!(
        component_round_trip(
            "post_processor",
            r#"{"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "a", "type_id": 0}},
                           {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                         {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {"a": {"id": "a", "ids": [0], "tokens": ["a"]}}}"#
        ),
        serde_json::json!({
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "a", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}}
            ],
            "special_tokens": {"a": {"id": "a", "ids": [0], "tokens": ["a"]}}
        })
    );
}

/// A `BertProcessing` is a frame, not a type, once it reaches the pipeline — so it comes back as the
/// template it always was. Different spelling, same ids, which is the writer's whole contract.
#[test]
fn a_bert_processing_becomes_its_template() {
    let written = component_round_trip(
        "post_processor",
        r#"{"type": "BertProcessing", "cls": ["a", 0], "sep": ["b", 1]}"#,
    );
    assert_eq!(written["type"], "TemplateProcessing");
    assert_eq!(
        written["single"],
        serde_json::json!([
            {"SpecialToken": {"id": "a", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "b", "type_id": 0}}
        ])
    );
    assert_eq!(
        written["special_tokens"]["a"]["ids"],
        serde_json::json!([0])
    );
    assert_eq!(
        written["special_tokens"]["b"]["ids"],
        serde_json::json!([1])
    );
}

/// `RobertaProcessing` pairs with a *doubled* separator: one placeholder standing for two ids. The
/// reconstructed name has to cover both, or the frame comes back a token short.
#[test]
fn a_doubled_special_token_run_keeps_both_ids() {
    let written = component_round_trip(
        "post_processor",
        r#"{"type": "RobertaProcessing", "cls": ["a", 0], "sep": ["b", 1],
            "trim_offsets": true, "add_prefix_space": true}"#,
    );
    let doubled = written["pair"][2]["SpecialToken"]["id"]
        .as_str()
        .expect("the doubled separator is a special-token piece")
        .to_string();
    assert_eq!(
        written["special_tokens"][&doubled]["ids"],
        serde_json::json!([1, 1]),
        "the doubled separator lost an id: {written}"
    );
}

#[test]
fn added_tokens_go_out_in_id_order_with_every_flag() {
    let written = field_of(
        &rewrite(&TINY_BPE.replace(
            r#""added_tokens": []"#,
            r#""added_tokens": [
                {"id": 5, "content": "<b>", "single_word": false, "lstrip": true,
                 "rstrip": false, "normalized": false, "special": true},
                {"id": 4, "content": "<a>", "single_word": true, "lstrip": false,
                 "rstrip": true, "normalized": true, "special": false}
            ]"#,
        )),
        "added_tokens",
    );
    assert_eq!(
        written,
        serde_json::json!([
            {"id": 4, "content": "<a>", "single_word": true, "lstrip": false,
             "rstrip": true, "normalized": true, "special": false},
            {"id": 5, "content": "<b>", "single_word": false, "lstrip": true,
             "rstrip": false, "normalized": false, "special": true}
        ]),
        "added tokens must go out in ascending id order: the reader replays them in that order, and \
         `add_tokens` reuses a model id when the token is already in the vocabulary"
    );
}

#[cfg(feature = "unigram")]
#[test]
fn a_unigram_model_keeps_its_scores_and_unk() {
    let written = field_of(
        &rewrite(&with_model(
            r#"{"type": "Unigram", "unk_id": 0, "byte_fallback": false,
                "vocab": [["<unk>", 0.0], ["a", -3.8403830528259277], ["b", -13.5321998596191]]}"#,
        )),
        "model",
    );
    assert_eq!(written["type"], "Unigram");
    assert_eq!(written["unk_id"], 0);
    assert_eq!(written["byte_fallback"], false);
    // `-3.8403830528259277` is the score `json.rs` documents as the one where our arithmetic and
    // `f64::from_str` disagree, so it is the interesting entry rather than a round number — and it
    // is why this asserts on *bits* rather than on text.
    //
    // The writer emits the shortest spelling of the value **our parser produced**, which is a
    // different double from the correctly-rounded reading of the same digits. Its shortest form is
    // `-3.840383052825928`: sixteen digits where the file had seventeen. Nothing is lost — the
    // seventeenth digit was describing a double the reader never built — but asserting the literal
    // here would be asserting the shortest form of the wrong number.
    let tokens = written["vocab"]
        .as_array()
        .expect("a Unigram vocab is an array");
    let expected = [
        0.0,
        crate::json::f64_from_literal("-3.8403830528259277"),
        crate::json::f64_from_literal("-13.5321998596191"),
    ];
    for (entry, want) in tokens.iter().zip(expected) {
        let literal = entry[1].to_string();
        let got = crate::json::f64_from_literal(&literal);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "the score written as {literal} reads back as {got}, not {want}"
        );
    }
    // And a score is still spelled as a float, never as a bare integer.
    assert_eq!(tokens[0][1].to_string(), "0.0");
}

#[cfg(feature = "wordpiece")]
#[test]
fn a_wordpiece_model_keeps_all_four_required_fields() {
    let written = field_of(
        &rewrite(&with_model(
            r###"{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1, "##b": 2}}"###,
        )),
        "model",
    );
    assert_eq!(
        written,
        serde_json::json!({
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "a": 1, "##b": 2}
        })
    );
}

/// A `Metaspace` normalizer only ever gets into a chain from a pre-tokenizer, and it is always last.
/// One anywhere else is a pipeline the writer must refuse rather than describe wrongly: written as a
/// `normalizer`, it would read back as a normalizer with the `Split` gone.
#[test]
fn a_stray_metaspace_normalizer_is_refused() {
    use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
    use tk_encode::normalizers::utils::Lowercase;
    use tk_encode::pipeline::PipelineNormalizer;

    let chain = vec![
        PipelineNormalizer::Metaspace(MetaspaceNormalizer::new('\u{2581}', true, false)),
        PipelineNormalizer::Lowercase(Lowercase),
    ];
    let mut out = super::writer::Out::new();
    let error = super::normalizers::write_normalizer(&mut out, &chain)
        .expect_err("a Metaspace that is not last cannot be written");
    assert!(
        error.to_string().contains("Metaspace"),
        "the error should name the component: {error}"
    );
}

/// [`to_json_file`] writes what [`to_json`] returns, and the file reads back as the same tokenizer.
///
/// Written into cargo's own `target` directory, under the test binary's name, so nothing lands in
/// `data/` — where another test's `.save(..)` already races the fixtures.
#[test]
fn to_json_file_writes_a_readable_config() {
    let tokenizer = from_json(TINY_BPE).expect("the tiny config reads");
    let path = std::env::temp_dir().join("tk_serialize_to_json_file_test.json");
    to_json_file(&tokenizer, &path).expect("writing a config to a file");
    let reread = from_json_file(&path).expect("the written file reads back");
    assert_eq!(
        ids(&tokenizer, "abab", false),
        ids(&reread, "abab", false),
        "a config written to a file did not read back as the same tokenizer"
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("read it back as text"),
        to_json(&tokenizer).expect("and write it again"),
        "the file does not hold exactly what `to_json` returns"
    );
    std::fs::remove_file(&path).expect("clean up");
}

/// The writer's output parses with an independent parser, not only with ours. Cheap, and it is what
/// catches a missing comma or an unescaped control character that `hifijson` happens to tolerate.
#[test]
fn the_output_is_valid_json_to_serde_too() {
    // The input spells them as JSON escapes; what matters is that the writer escapes them again on
    // the way out, because a raw control character is not legal JSON and `hifijson` refuses one.
    let config = with_component(
        "normalizer",
        r#"{"type": "Replace", "pattern": {"String": "\u0001\t\"\\"}, "content": "\u001f"}"#,
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&rewrite(&config)).expect("serde_json parses the writer's output");
    assert_eq!(
        parsed["normalizer"]["pattern"]["String"], "\u{1}\t\"\\",
        "control characters and quotes did not survive escaping"
    );
    assert_eq!(parsed["normalizer"]["content"], "\u{1f}");
}
