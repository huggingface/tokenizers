//! The writer's tests. The gate is [`round_trip_preserves_ids_on_every_real_config`]: the contract
//! is ids, not bytes, because the pipeline is a lowered form of the file. The rest say *where*.

use super::writer::*;
use super::*;
use crate::from_json::from_json;
use crate::json::Json;
use crate::vendored::f64_from_literal;

const BPE_MODEL: &str = r#"{"type": "BPE", "byte_level": false,
    "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3}, "merges": [["a", "b"], ["ab", "ab"]]}"#;

/// A whole config: every component `null` and [`BPE_MODEL`], unless `slots` names one.
fn config(slots: &[(&str, &str)]) -> String {
    let slot = |name: &str, default: &str| -> String {
        slots
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(default, |(_, json)| json)
            .to_string()
    };
    format!(
        r#"{{"version": "2.0", "added_tokens": {}, "normalizer": {}, "pre_tokenizer": {},
            "post_processor": {}, "decoder": {}, "model": {}}}"#,
        slot("added_tokens", "[]"),
        slot("normalizer", "null"),
        slot("pre_tokenizer", "null"),
        slot("post_processor", "null"),
        slot("decoder", "null"),
        slot("model", BPE_MODEL),
    )
}

fn rewrite(text: &str) -> String {
    to_json(&from_json(text).expect("the config reads")).expect("a config that reads should write")
}

/// One field of a written config, through `serde_json` -- an *independent* parser, which is what
/// makes "valid JSON" an assertion rather than a check against ourselves.
fn field_of(written: &str, field: &str) -> serde_json::Value {
    let parsed: serde_json::Value =
        serde_json::from_str(written).expect("the writer emits valid JSON");
    parsed[field].clone()
}

/// What `slot` becomes after a read and a write.
fn written(slot: &str, json: &str) -> serde_json::Value {
    field_of(&rewrite(&config(&[(slot, json)])), slot)
}

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("a test expectation is valid JSON")
}

fn ids(tokenizer: &PipelineTokenizer, text: &str, specials: bool) -> Vec<u32> {
    let encoded = tokenizer
        .encode(text, specials)
        .wait()
        .expect("encoding a text");
    encoded
        .iter()
        .flat_map(|e| e.ids())
        .map(|token| token.id())
        .collect()
}

/// Every `data/*.json` the reader accepts, built. A refusal is a skip; `tokenizer.json` is another
/// test binary's output, written in parallel with this one.
fn fixtures() -> Vec<(String, PipelineTokenizer)> {
    let Ok(entries) = std::fs::read_dir("../data") else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name != "tokenizer.json")
        })
        .collect();
    paths.sort();
    paths
        .iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_string_lossy().to_string();
            Some((name, from_json(&std::fs::read_to_string(path).ok()?).ok()?))
        })
        .collect()
}

/// One text per encode regime rather than one per language, plus the special-token markers that
/// catch an added-token replay in the wrong order.
#[rustfmt::skip]
const TEXTS: &[&str] = &[
    " the quick brown fox jumps over the lazy dog",
    "def foo(bar):\n    return bar + 1\n",
    " 语言模型 mixed with ASCII and ελληνικά",
    "الْعَرَبِيَّة и русский текст",
    "3.14159 and 1e-9 and 0007",
    "[CLS] <s> <|endoftext|> [SEP] </s> <unk> [MASK]",
    "tabs\tand\nnewlines\r\nand  double  spaces", "   ", "",
];

/// Read a real config, write it, read it back, encode with both. A *writer* refusal is a failure,
/// not a skip: everything the reader builds, the writer must describe.
// TODO(pr3): every fixture in `data/` is a 1.0 file, so this has no input until tk-convert can
// produce 2.0 ones. Port it there, with the per-score float sweep git remembers.
#[ignore = "needs 2.0 fixtures from tk-convert (pr3)"]
#[test]
fn round_trip_preserves_ids_on_every_real_config() {
    let mut configs = 0usize;
    for (name, before) in fixtures() {
        let text = to_json(&before)
            .unwrap_or_else(|e| panic!("{name}: the reader built it, the writer cannot: {e}"));
        let after = from_json(&text)
            .unwrap_or_else(|e| panic!("{name}: its own reader refuses the output: {e}"));
        configs += 1;
        for text in TEXTS {
            for specials in [false, true] {
                assert_eq!(
                    ids(&before, text, specials),
                    ids(&after, text, specials),
                    "{name}: ids moved across a round trip (specials={specials}) on {text:?}"
                );
            }
        }
    }
    assert!(configs >= 1, "every config in ../data was skipped");
}

/// Merges that *compete*: `a+b` and `b+c` both apply to `abc`, so rank alone picks the winner.
/// [`BPE_MODEL`] cannot serve -- its merges form a chain, which offers no choice.
const COMPETING_BPE: &str = r#"{"type": "BPE", "byte_level": false,
    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4}, "merges": [["a", "b"], ["b", "c"]]}"#;

/// The gate has teeth. Perturbing the *written config* rather than the writer is what keeps this
/// from rotting: it shows the comparison above is sensitive to a wrong merge order, whoever made it.
#[test]
fn reversing_the_written_merges_moves_ids() {
    let tokenizer = from_json(&config(&[("model", COMPETING_BPE)])).expect("it reads");
    let mut parsed: serde_json::Value =
        serde_json::from_str(&to_json(&tokenizer).expect("and writes")).expect("valid JSON");
    assert_eq!(
        parsed["model"]["merges"],
        json(r#"[["a","b"],["b","c"]]"#),
        "not rank order"
    );
    parsed["model"]["merges"]
        .as_array_mut()
        .expect("merges")
        .reverse();
    let perturbed = from_json(&parsed.to_string()).expect("the perturbed config still reads");

    // `a+b` outranks `b+c`, so `abc` is `ab` + `c`. Reversed, `b+c` wins: `a` + `bc`.
    assert_eq!(ids(&tokenizer, "abc", false), vec![3, 2]);
    assert_eq!(
        ids(&perturbed, "abc", false),
        vec![0, 4],
        "reversing the merge order left the ids alone, so the gate compares nothing"
    );

    // And the two weak perturbations, so nobody rediscovers that they are inert: popping the last
    // merge, and reversing a *chain*. Both look exactly like a dead gate.
    let chained = from_json(&config(&[])).expect("the chain config reads");
    let text = to_json(&chained).expect("and writes");
    assert_eq!(ids(&chained, "abab", false), vec![3], "`abab` merges up");
    for pop in [true, false] {
        let mut parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let merges = parsed["model"]["merges"].as_array_mut().expect("merges");
        if pop {
            merges.pop();
        } else {
            merges.reverse();
        }
        let weak = from_json(&parsed.to_string()).expect("a weak perturbation still reads");
        assert_eq!(
            ids(&weak, "abab", false),
            vec![3],
            "pop={pop} moved an id after all"
        );
    }
}

/// `(slot, json)` for every component whose canonical spelling is what went in: read it, write it,
/// and the same object comes back.
#[rustfmt::skip]
const IDEMPOTENT: &[(&str, &str)] = &[
    ("normalizer",     r#"{"type": "Lowercase"}"#),
    ("normalizer",     r#"{"type": "Prepend", "prepend": "_"}"#),
    ("normalizer",     r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#),
    ("normalizer",     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": "always", "drop_whitespace": false}"#),
    ("normalizer",     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": "first", "drop_whitespace": false}"#),
    ("normalizer",     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": "never", "drop_whitespace": false}"#),
    // `drop_whitespace` is one flag, never the legacy `Sequence[WhitespaceSplit, Metaspace]`.
    ("normalizer",     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": "always", "drop_whitespace": true}"#),
    ("pre_tokenizer",  r#"{"type": "Digits", "individual_digits": true}"#),
    ("pre_tokenizer",  r#"{"type": "Whitespace"}"#),
    ("pre_tokenizer",  r#"{"type": "WhitespaceSplit"}"#),
    ("pre_tokenizer",  r#"{"type": "BertPreTokenizer"}"#),
    ("pre_tokenizer",  r#"{"type": "Punctuation", "behavior": "Removed"}"#),
    ("pre_tokenizer",  r#"{"type": "CharDelimiterSplit", "delimiter": "-"}"#),
    ("pre_tokenizer",  r#"{"type": "FixedLength", "length": 7}"#),
    ("pre_tokenizer",  r#"{"type": "Split", "pattern": {"String": "-"}, "behavior": "MergedWithNext", "invert": false}"#),
    ("pre_tokenizer",  r#"{"type": "Sequence", "pretokenizers": [{"type": "Whitespace"}, {"type": "Digits", "individual_digits": false}]}"#),
    ("decoder",        r#"{"type": "Fuse"}"#),
    ("decoder",        r#"{"type": "ByteFallback"}"#),
    ("decoder",        r#"{"type": "Strip", "content": "_", "start": 1, "stop": 0}"#),
    ("decoder",        r#"{"type": "BPEDecoder", "suffix": "</w>"}"#),
    ("decoder",        r###"{"type": "WordPiece", "prefix": "##", "cleanup": true}"###),
    ("decoder",        r#"{"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}"#),
    ("decoder",        r#"{"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}"#),
    ("post_processor", r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}], "pair": [{"seq": "A"}, {"seq": "B", "type_id": 1}]}"#),
];

/// `(slot, in, out)`: what the pipeline folded or dropped. An empty `Sequence` disappears, a nested
/// one flattens, a decoder reads past `ByteLevel`'s flags and `Metaspace`'s `split` (but keeps
/// `prepend_scheme: first`), and Bert/Roberta processing are frames, so both become one template.
#[rustfmt::skip]
const REWRITTEN: &[(&str, &str, &str)] = &[
    ("normalizer", r#"{"type": "Sequence", "normalizers": []}"#, "null"),
    ("normalizer", r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"},
        {"type": "Sequence", "normalizers": [{"type": "Strip", "strip_left": true, "strip_right": false}]}]}"#,
                   r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"},
        {"type": "Strip", "strip_left": true, "strip_right": false}]}"#),
    ("decoder", r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
                r#"{"type": "ByteLevel"}"#),
    ("decoder", r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}"#,
                r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}"#),
    ("post_processor", r#"{"type": "BertProcessing", "cls": ["a", 0], "sep": ["b", 1]}"#,
        r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}, {"ids": [1]}],
            "pair": [{"ids": [0]}, {"seq": "A"}, {"ids": [1]}, {"seq": "B", "type_id": 1}, {"ids": [1], "type_id": 1}]}"#),
    ("post_processor", r#"{"type": "RobertaProcessing", "cls": ["a", 0], "sep": ["b", 1], "trim_offsets": true, "add_prefix_space": true}"#,
        r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}, {"ids": [1]}],
            "pair": [{"ids": [0]}, {"seq": "A"}, {"ids": [1, 1]}, {"seq": "B"}, {"ids": [1]}]}"#),
    // Ascending id order is load-bearing: the reader replays added tokens in that order, and
    // `add_tokens` reuses a model id when the token is already in the vocabulary.
    ("added_tokens", r#"[{"id": 5, "content": "<b>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true},
        {"id": 4, "content": "<a>", "single_word": true, "lstrip": false, "rstrip": true, "normalized": true, "special": false}]"#,
        r#"[{"id": 4, "content": "<a>", "single_word": true, "lstrip": false, "rstrip": true, "normalized": true, "special": false},
        {"id": 5, "content": "<b>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true}]"#),
];

/// `strip_accents: null` means "decide from `lowercase`", which is why the reader requires the key.
#[cfg(feature = "normalizers")]
#[rustfmt::skip]
const BERT: &[(&str, &str)] = &[
    ("normalizer", r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}"#),
];
#[cfg(not(feature = "normalizers"))]
const BERT: &[(&str, &str)] = &[];

/// All four fields are required, so none may go missing on the way out.
#[cfg(feature = "wordpiece")]
#[rustfmt::skip]
const WORDPIECE: &[(&str, &str)] = &[
    ("model", r###"{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1, "##b": 2}}"###),
];
#[cfg(not(feature = "wordpiece"))]
const WORDPIECE: &[(&str, &str)] = &[];

#[test]
fn components_round_trip_to_their_canonical_spelling() {
    for (slot, spelling) in IDEMPOTENT.iter().chain(BERT).chain(WORDPIECE) {
        assert_eq!(
            written(slot, spelling),
            json(spelling),
            "{slot}: {spelling}"
        );
    }
    for (slot, input, expected) in REWRITTEN {
        assert_eq!(written(slot, input), json(expected), "{slot}: {input}");
    }
}

#[test]
fn the_canonical_shape_is_tagged_versioned_and_null_where_absent() {
    let text = rewrite(&config(&[
        (
            "normalizer",
            r#"{"type": "MetaspaceNormalizer", "replacement": "▁",
            "prepend": "always", "drop_whitespace": false}"#,
        ),
        ("pre_tokenizer", r#"{"type": "Whitespace"}"#),
        ("decoder", r#"{"type": "Fuse"}"#),
    ]));
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(parsed["version"], "2.0");
    // The reader still tolerates an untagged model; the writer must never make one.
    for field in ["normalizer", "pre_tokenizer", "decoder", "model"] {
        assert!(
            parsed[field]["type"].as_str().is_some(),
            "`{field}` has no `type`"
        );
    }
    // Pairs in rank order, never the legacy `"a b"`, which is ambiguous when a token has a space.
    assert_eq!(
        parsed["model"]["merges"],
        json(r#"[["a","b"],["ab","ab"]]"#)
    );
    for legacy in ["add_prefix_space", "prepend_scheme", "split"] {
        assert!(
            parsed["normalizer"].get(legacy).is_none(),
            "wrote `{legacy}`"
        );
    }

    // A pass-through frame is what "no post-processor" lowers to, so it goes back out as absent.
    let bare = rewrite(&config(&[]));
    for field in ["normalizer", "pre_tokenizer", "decoder", "post_processor"] {
        assert_eq!(field_of(&bare, field), serde_json::Value::Null, "{field}");
    }
    assert_eq!(field_of(&bare, "truncation"), serde_json::Value::Null);
    assert_eq!(field_of(&bare, "padding"), serde_json::Value::Null);
}

/// On *bits*, not text: the writer emits the shortest spelling of the double **our parser
/// produced**, which for `-3.8403830528259277` is not the correctly-rounded reading of those digits.
#[cfg(feature = "unigram")]
#[test]
fn a_unigram_model_keeps_its_scores_and_unk() {
    let model = written(
        "model",
        r#"{"type": "Unigram", "unk_id": 0, "byte_fallback": false,
            "vocab": [["<unk>", 0.0], ["a", -3.8403830528259277], ["b", -13.5321998596191]]}"#,
    );
    assert_eq!(model["type"], "Unigram");
    assert_eq!(model["unk_id"], 0);
    assert_eq!(model["byte_fallback"], false);
    let vocab = model["vocab"]
        .as_array()
        .expect("a Unigram vocab is an array");
    for (entry, digits) in vocab
        .iter()
        .zip(["0.0", "-3.8403830528259277", "-13.5321998596191"])
    {
        let literal = entry[1].to_string();
        let got = crate::vendored::f64_from_literal(&literal);
        let want = crate::vendored::f64_from_literal(digits);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{literal} reads back as {got}"
        );
    }
    assert_eq!(vocab[0][1].to_string(), "0.0");
}

/// Cheap, and it catches an unescaped control character that `hifijson` happens to tolerate: the
/// input spells them as JSON escapes, and the writer has to escape them again on the way out.
#[test]
fn the_output_is_valid_json_to_serde_too() {
    let normalizer = written(
        "normalizer",
        r#"{"type": "Replace", "pattern": {"String": "\u0001\t\"\\"}, "content": "\u001f"}"#,
    );
    assert_eq!(normalizer["pattern"]["String"], "\u{1}\t\"\\");
    assert_eq!(normalizer["content"], "\u{1f}");
}

// ---- the emitter itself ----------------------------------------------------------------------
/// Round-trip through the *public* accessor, not just `f64_from_literal`, so the test covers
/// the path a reader really takes.
fn reads_back_as(literal: &str) -> f64 {
    Json::parse(literal)
        .expect("the writer emits parseable JSON")
        .as_f64()
        .expect("a number literal reads as an f64")
}

#[test]
fn floats_round_trip_through_our_own_parser() {
    for value in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        // Two real Unigram scores rather than round numbers, both read through the parser they
        // have to survive: `-3.8403830528259277` is the one
        // `matches_serde_not_from_str_on_a_real_unigram_score` pins, where the value our parser
        // gives is a ULP off the correctly-rounded one.
        f64_from_literal("-13.5321998596191"),
        f64_from_literal("-3.8403830528259277"),
        f64::MIN_POSITIVE,
        f64::MAX,
        1e-300,
        1e300,
        std::f64::consts::PI,
    ] {
        let literal = float_literal(value).expect("every finite float has a spelling");
        assert_eq!(
            reads_back_as(&literal).to_bits(),
            value.to_bits(),
            "{value:?} was written as {literal}, which reads back as {}",
            reads_back_as(&literal)
        );
    }
}

#[test]
fn non_finite_numbers_are_refused_rather_than_mangled() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(float_literal(value).is_err(), "{value:?} has no JSON form");
    }
}

/// Every escape `hifijson` insists on, plus the ones it does not but a reader would find
/// surprising. A raw control character is not legal JSON, so the `\u00XX` arm is not cosmetic.
#[test]
fn strings_escape_what_json_requires() {
    let mut out = Out::new();
    out.str("a\"b\\c\nd\te\rf\u{8}g\u{c}h\u{1}i\u{1f}j");
    let written = out.finish();
    // Spelled as an ordinary string literal rather than a raw one, so that every escape the
    // writer is expected to produce is visible here instead of being a raw control byte.
    assert_eq!(
        written, "\"a\\\"b\\\\c\\nd\\te\\rf\\bg\\fh\\u0001i\\u001fj\"",
        "escaping changed"
    );
    assert_eq!(
        Json::parse(&written)
            .expect("escaped output parses")
            .as_str(),
        Some("a\"b\\c\nd\te\rf\u{8}g\u{c}h\u{1}i\u{1f}j"),
        "the escaped form does not read back as the original"
    );
}

/// Non-ASCII goes out raw, which is what keeps a byte-level vocabulary readable.
#[test]
fn non_ascii_is_not_escaped() {
    let mut out = Out::new();
    out.str("Ġthe▁世界");
    let written = out.finish();
    assert_eq!(written, "\"Ġthe▁世界\"");
    assert_eq!(
        Json::parse(&written).expect("parses").as_str(),
        Some("Ġthe▁世界")
    );
}

#[test]
fn containers_get_their_commas() {
    let mut out = Out::new();
    out.obj_open();
    out.type_tag("Demo");
    out.field_bool("flag", true);
    out.field_u32("id", 7);
    out.field_null("nothing");
    out.key("list");
    out.arr_open();
    out.u32(1);
    out.u32(2);
    out.obj_open();
    out.field_str("k", "v");
    out.obj_close();
    out.arr_close();
    out.obj_close();
    let written = out.finish();
    assert_eq!(
        written,
        r#"{"type":"Demo","flag":true,"id":7,"nothing":null,"list":[1,2,{"k":"v"}]}"#
    );
    // And it is a document our own parser accepts, which is the property that matters.
    let parsed = Json::parse(&written).expect("emitted JSON parses");
    assert_eq!(parsed.type_tag(), Some("Demo"));
}

#[test]
fn integers_are_written_without_a_fraction() {
    let mut out = Out::new();
    out.arr_open();
    out.u32(0);
    out.u32(u32::MAX);
    out.usize(50256);
    out.arr_close();
    assert_eq!(out.finish(), "[0,4294967295,50256]");
}

/// `role_to_token` is what lets a `tokenizer.json` carry the special-token metadata that used to
/// need a `tokenizer_config.json`, so it has to survive a read and a write unchanged. Declaring
/// none writes `null` rather than an empty object.
#[test]
fn role_to_token_survives_a_round_trip() {
    let declared = r#"{"eos_token": "</s>", "bos_token": "<s>", "pad_token": "<pad>"}"#;
    let text = format!(
        r#"{{"version": "2.0", "role_to_token": {declared}, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": null, "model": {BPE_MODEL}}}"#
    );

    let tokenizer = from_json(&text).expect("the config reads");
    assert_eq!(tokenizer.get_token_for_role("eos_token"), Some("</s>"));
    assert_eq!(tokenizer.get_token_for_role("nonexistent"), None);

    let written = to_json(&tokenizer).expect("a config that reads should write");
    assert_eq!(field_of(&written, "role_to_token"), json(declared));

    // Absent means no roles, and comes back out as `null`.
    let bare = rewrite(&config(&[]));
    assert_eq!(field_of(&bare, "role_to_token"), serde_json::Value::Null);
    assert!(
        from_json(&config(&[]))
            .expect("the config reads")
            .get_role_to_token()
            .is_empty()
    );
}
