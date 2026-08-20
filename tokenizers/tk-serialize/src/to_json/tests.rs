//! The writer's tests. The one that matters is [`round_trip_preserves_ids_on_every_real_config`].
//!
//! The contract this writer offers is *ids*, not bytes — the pipeline is a lowered form and cannot
//! reproduce the file it came from (see the module docs). So the gate is: read a real config, write
//! it, read that back, and encode with both. Anything the writer gets wrong about a component shows
//! up as a different id, which is the only thing a caller can observe.
//!
//! Everything else here covers what that gate cannot. The float sweep pins the single value most
//! likely to shift silently and least likely to show up in a fixed corpus; the canonical-spelling
//! test pins the shapes the *reader* still tolerates, which the round trip therefore cannot see; and
//! the component table covers the shapes no `data/` fixture contains.

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

/// [`TINY_BPE`] with one top-level component filled in. Every such field is `null` there, so a plain
/// textual replace is unambiguous.
fn with_component(field: &str, json: &str) -> String {
    TINY_BPE.replace(
        &format!(r#""{field}": null"#),
        &format!(r#""{field}": {json}"#),
    )
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

/// The perturbation that proves the round-trip gate has teeth: reverse the merge list of a real
/// fixture and the ids must move.
///
/// Applied to the *written config* rather than to the writer, so the test cannot rot when the writer
/// changes: what it demonstrates is that the comparison
/// [`round_trip_preserves_ids_on_every_real_config`] performs is sensitive to a wrong merge order,
/// whoever produced it.
///
/// Finding a perturbation that actually bites took two attempts, and both failures are worth
/// recording so nobody has to rediscover them:
///
/// - `merges.pop()` does nothing. The lowest-priority pair is the one least likely to be reached at
///   all, and on a byte-level model the byte-level fold can emit a vocabulary entry without merging
///   anything.
/// - `merges.reverse()` does nothing *either*, as long as the merges form a chain. [`TINY_BPE`]'s
///   do — `a+b` makes `ab`, then `ab+ab` makes `abab` — and BPE applies whatever pair is in front of
///   it, so a chain never offers a choice and reversing it changes no token.
///
/// What bites is reversing merges that **compete**: two pairs applicable to the same position, where
/// the ranks are the only tie-break. A real vocabulary is full of those, which is why the fixtures
/// are the right place for this and a four-token toy was not.
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

/// A legacy config in, a canonical one out. `gpt2`-style files ship the old spellings, and the point
/// of the writer is that reading one and writing it produces the modern one.
///
/// This is the contract the round-trip gate cannot see. Every clause below is something the *reader*
/// still tolerates — an untagged model, `"a b"` merges, an omitted key — so writing it back the old
/// way would read back as the same tokenizer and move no id at all.
#[test]
fn a_legacy_config_is_written_canonically() {
    let legacy = r#"{
        "version": "1.0",
        "added_tokens": [],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": {"type": "Fuse"},
        "model": {
            "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
            "merges": ["a b", "ab ab"]
        }
    }"#;
    let written = rewrite(legacy);
    let parsed: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");

    // Every component carries an explicit `"type"`, the model included -- which is the one the
    // reader would have accepted without.
    for field in ["normalizer", "pre_tokenizer", "decoder", "model"] {
        assert!(
            parsed[field]["type"].as_str().is_some(),
            "`{field}` was written without a `type`: {}",
            parsed[field]
        );
    }
    assert_eq!(parsed["model"]["type"], "BPE");
    assert_eq!(parsed["version"], "1.0");

    // Merges go out as `["a", "b"]` pairs in rank order, never as the legacy `"a b"` string --
    // which is ambiguous exactly when a token contains a space, and is why the pair form became
    // canonical.
    assert_eq!(
        parsed["model"]["merges"],
        serde_json::json!([["a", "b"], ["ab", "ab"]]),
        "merges are not canonical pairs in rank order"
    );

    // And an absent component is spelled `null` rather than omitted. `get_some` treats the two as
    // the same document, so a present key is the writer saying it considered the field.
    let bare = rewrite(TINY_BPE);
    for field in [
        "normalizer",
        "pre_tokenizer",
        "decoder",
        // A pass-through frame is what "no post-processor" lowers to, so it goes back out as absent.
        "post_processor",
        // Neither of these is part of a pipeline, and both have always been spelled out.
        "truncation",
        "padding",
    ] {
        assert_eq!(
            field_of(&bare, field),
            serde_json::Value::Null,
            "`{field}` should be written as an explicit null"
        );
    }
}

// ---- the components no fixture covers --------------------------------------------------------
//
// The round-trip gate reads and writes every `data/*.json`, so a component that appears in one is
// already covered — written wrongly, it reads back as a different pipeline and the ids move. What is
// left is the components and spellings the Hub's files happen not to contain, and those have no gate
// but this one.

/// Every component shape absent from `data/`, written and read back as itself.
///
/// One table rather than one test per component: the assertion is identical in each case — this
/// config in, that JSON out — and the interesting content is *which* shapes are on the list, which a
/// table shows and a dozen functions bury. Each entry says why no fixture reaches it.
#[test]
fn components_absent_from_the_fixtures_keep_their_fields() {
    for (field, input, expected) in [
        // A `Sequence` nested inside a `Sequence`. The pipeline holds only the flattening, so that
        // is what comes out; both spellings read back as the same chain. A `Strip` with only one
        // side set is absent too -- every fixture sets both.
        (
            "normalizer",
            r#"{"type": "Sequence", "normalizers": [
                {"type": "Lowercase"},
                {"type": "Sequence", "normalizers": [
                    {"type": "Strip", "strip_left": true, "strip_right": false}
                ]},
                {"type": "Prepend", "prepend": "_"}
            ]}"#,
            serde_json::json!({
                "type": "Sequence",
                "normalizers": [
                    {"type": "Lowercase"},
                    {"type": "Strip", "strip_left": true, "strip_right": false},
                    {"type": "Prepend", "prepend": "_"}
                ]
            }),
        ),
        // Four pre-tokenizers no fixture in `data/` uses at all.
        (
            "pre_tokenizer",
            r#"{"type": "Digits", "individual_digits": true}"#,
            serde_json::json!({"type": "Digits", "individual_digits": true}),
        ),
        (
            "pre_tokenizer",
            r#"{"type": "Punctuation", "behavior": "Removed"}"#,
            serde_json::json!({"type": "Punctuation", "behavior": "Removed"}),
        ),
        (
            "pre_tokenizer",
            r#"{"type": "CharDelimiterSplit", "delimiter": "-"}"#,
            serde_json::json!({"type": "CharDelimiterSplit", "delimiter": "-"}),
        ),
        (
            "pre_tokenizer",
            r#"{"type": "FixedLength", "length": 7}"#,
            serde_json::json!({"type": "FixedLength", "length": 7}),
        ),
        // Two decoders no fixture uses. A decoder does not affect an id at all, so the round-trip
        // gate cannot see one either way; what covers the rest is `tk-convert`'s decode oracle, and
        // that only reaches the decoders the fixtures ship.
        (
            "decoder",
            r#"{"type": "BPEDecoder", "suffix": "</w>"}"#,
            serde_json::json!({"type": "BPEDecoder", "suffix": "</w>"}),
        ),
        (
            "decoder",
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
            // A decoder keeps all three prepend schemes, and `split: false` with them: nothing here
            // has to be expressible as a normalizer, so `first` survives where the pre-tokenizer's
            // does not. Every fixture's Metaspace decoder spells `always` and `split: true`.
            "decoder",
            r#"{"type": "Metaspace", "replacement": "▁",
                "prepend_scheme": "first", "split": false}"#,
            serde_json::json!({
                "type": "Metaspace",
                "replacement": "\u{2581}",
                "prepend_scheme": "first",
                "split": false
            }),
        ),
    ] {
        assert_eq!(component_round_trip(field, input), expected, "{input}");
    }
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

/// A frame processor is a frame, not a type, once it reaches the pipeline — so it comes back as the
/// template it always was. Different spelling, same ids, which is the writer's whole contract. The
/// names are rebuilt from the ids, so this checks that the rebuilt table resolves to the same frame.
///
/// Neither half is reachable from the fixtures. `BertProcessing` appears in no `data/*.json` at all,
/// and the `pair` template appears in plenty but is never *exercised*: the round-trip gate encodes
/// single sequences, so nothing there ever applies a pair frame.
#[test]
fn a_frame_processor_becomes_its_template() {
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

    // `RobertaProcessing` pairs with a *doubled* separator: one placeholder standing for two ids.
    // The reconstructed name has to cover both, or the frame comes back a token short.
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

/// The hybrid layout over the fixtures that make it matter, and the size it buys.
///
/// The exact-output test below pins the shape on a four-token toy; this one checks the *property*
/// that shape exists for, on real vocabularies: the written file is far smaller than the fully
/// indented one on disk, and the skeleton stays something a human could scroll. Both assertions
/// catch the same regression -- a `begin_compact` lost from a bulk collection -- which would take
/// `gemma-4.json` from 52 lines to about 1.5 million and from 13.6 MB to 32.2 MB, and which no other
/// test here can see, because they all parse the output before comparing anything.
#[test]
fn the_layout_shrinks_the_real_configs_and_keeps_them_readable() {
    let mut measured = 0usize;
    for f in [
        "gemma-4.json",
        "gpt-oss.json",
        "xlmr.json",
        "glm-5.2.json",
        "mistral-small-4.json",
        "llama-3-tokenizer.json",
        "deepseek-v4.json",
    ] {
        let path = format!("../data/{f}");
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let on_disk = std::fs::metadata(&path).unwrap().len();
        let Ok(tok) = from_json_file(&path) else {
            eprintln!("  skip {f}");
            continue;
        };
        // `to_json` unqualified: the crate-root re-export is behind `serialize`, and these tests
        // also build in a plain `deserialize` test build, where only the module's own item exists.
        let out = to_json(&tok).unwrap();
        let lines = out.lines().count();
        let longest = out.lines().map(str::len).max().unwrap_or(0);
        eprintln!(
            "  {f:<38} disk {:>6.2} MB   hybrid {:>6.2} MB   {:>3.0}% smaller   {lines} lines, longest {:.2} MB",
            on_disk as f64 / 1e6,
            out.len() as f64 / 1e6,
            100.0 * (1.0 - out.len() as f64 / on_disk as f64),
            longest as f64 / 1e6,
        );
        assert!(
            (out.len() as u64) < on_disk,
            "{f}: the hybrid form is {} bytes against {on_disk} on disk, so the whitespace saving \
             has gone",
            out.len()
        );
        // Generous on purpose: the widest skeleton here is xlmr's 121 lines, and indenting a
        // vocabulary instead would be six figures. Anything between is a shape nobody chose.
        assert!(
            lines < 1_000,
            "{f}: the skeleton is {lines} lines, so a bulk collection is being indented"
        );
        measured += 1;
    }
    assert!(
        measured >= 1,
        "no fixture was measured; `make data models` fetches them"
    );
}

/// The layout, pinned exactly, because it is the one contract no other test here can see: every
/// gate above parses the output and compares ids, and whitespace survives all of them.
///
/// The shape is a hybrid. The skeleton is indented, because that is the part a human reads; the bulk
/// collections -- `added_tokens`, `model.vocab`, `model.merges`, and a Unigram vocab's `[token,
/// score]` pairs -- carry no whitespace at all, because indenting those is what makes a
/// `tokenizer.json` enormous: fully indented, `gemma-4.json` is 32.2 MB against 13.6 MB compact, and
/// the hybrid buys a 52-line readable skeleton for 271 of those bytes.
///
/// `added_tokens` sits after `decoder` rather than up at the top, so both fat sections are at the
/// end and everything a human reads comes first.
#[test]
fn the_layout_is_an_indented_skeleton_around_compact_bulk() {
    assert_eq!(
        rewrite(TINY_BPE),
        r#"{
  "version":"1.0",
  "truncation":null,
  "padding":null,
  "normalizer":null,
  "pre_tokenizer":null,
  "post_processor":null,
  "decoder":null,
  "added_tokens":[],
  "model":{
    "type":"BPE",
    "dropout":null,
    "unk_token":null,
    "continuing_subword_prefix":null,
    "end_of_word_suffix":null,
    "fuse_unk":false,
    "byte_fallback":false,
    "ignore_merges":true,
    "vocab":{"a":0,"b":1,"ab":2,"abab":3},
    "merges":[["a","b"],["ab","ab"]]
  }
}"#
    );
}
