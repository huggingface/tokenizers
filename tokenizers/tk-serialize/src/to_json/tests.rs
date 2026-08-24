//! The writer's tests. The gate is [`round_trip_preserves_ids_on_every_real_config`]: the contract
//! is ids, not bytes, because the pipeline is a lowered form of the file. The rest say *where*.

use super::*;
use crate::from_json::from_json;

const BPE_MODEL: &str = r#"{"type": "BPE", "byte_level": false,
    "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3}, "merges": [["a", "b"], ["ab", "ab"]]}"#;

/// A whole config: every component `null` and [`BPE_MODEL`], unless `slots` names one.
fn config(slots: &[(&str, &str)]) -> String {
    let slot = |name: &str, default: &str| -> String {
        slots.iter().find(|(n, _)| *n == name).map_or(default, |(_, json)| json).to_string()
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
    let encoded = tokenizer.encode(text, specials).wait().expect("encoding a text");
    encoded.iter().flat_map(|e| e.ids()).map(|token| token.id()).collect()
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
        .filter(|path| path.file_name().is_some_and(|name| name != "tokenizer.json"))
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
    assert_eq!(parsed["model"]["merges"], json(r#"[["a","b"],["b","c"]]"#), "not rank order");
    parsed["model"]["merges"].as_array_mut().expect("merges").reverse();
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
        if pop { merges.pop(); } else { merges.reverse(); }
        let weak = from_json(&parsed.to_string()).expect("a weak perturbation still reads");
        assert_eq!(ids(&weak, "abab", false), vec![3], "pop={pop} moved an id after all");
    }
}

/// `(slot, json)` for every component whose canonical spelling is what went in: read it, write it,
/// and the same object comes back.
#[rustfmt::skip]
const IDEMPOTENT: &[(&str, &str)] = &[
    ("normalizer",     r#"{"type": "Lowercase"}"#),
    ("normalizer",     r#"{"type": "Prepend", "prepend": "_"}"#),
    ("normalizer",     r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#),
    ("normalizer",     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": true, "drop_whitespace": false}"#),
    // `drop_whitespace` is one flag, never the legacy `Sequence[WhitespaceSplit, Metaspace]`.
    ("normalizer",     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": true, "drop_whitespace": true}"#),
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
        assert_eq!(written(slot, spelling), json(spelling), "{slot}: {spelling}");
    }
    for (slot, input, expected) in REWRITTEN {
        assert_eq!(written(slot, input), json(expected), "{slot}: {input}");
    }
}

#[test]
fn the_canonical_shape_is_tagged_versioned_and_null_where_absent() {
    let text = rewrite(&config(&[
        ("normalizer", r#"{"type": "MetaspaceNormalizer", "replacement": "▁",
            "prepend": true, "drop_whitespace": false}"#),
        ("pre_tokenizer", r#"{"type": "Whitespace"}"#),
        ("decoder", r#"{"type": "Fuse"}"#),
    ]));
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(parsed["version"], "2.0");
    // The reader still tolerates an untagged model; the writer must never make one.
    for field in ["normalizer", "pre_tokenizer", "decoder", "model"] {
        assert!(parsed[field]["type"].as_str().is_some(), "`{field}` has no `type`");
    }
    // Pairs in rank order, never the legacy `"a b"`, which is ambiguous when a token has a space.
    assert_eq!(parsed["model"]["merges"], json(r#"[["a","b"],["ab","ab"]]"#));
    for legacy in ["add_prefix_space", "prepend_scheme", "split"] {
        assert!(parsed["normalizer"].get(legacy).is_none(), "wrote `{legacy}`");
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
    let vocab = model["vocab"].as_array().expect("a Unigram vocab is an array");
    for (entry, digits) in vocab.iter().zip(["0.0", "-3.8403830528259277", "-13.5321998596191"]) {
        let literal = entry[1].to_string();
        let got = crate::vendored::f64_from_literal(&literal);
        let want = crate::vendored::f64_from_literal(digits);
        assert_eq!(got.to_bits(), want.to_bits(), "{literal} reads back as {got}");
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
