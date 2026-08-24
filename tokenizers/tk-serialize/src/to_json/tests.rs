//! The writer's tests. The gate is [`round_trip_preserves_ids_on_every_real_config`]: the contract
//! is ids, not bytes, because the pipeline is a lowered form of the file. The rest say *where*.

use super::*;
use crate::from_json::from_json;

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

/// One top-level field of a written config, through `serde_json` -- an *independent* parser, which
/// is what makes "the writer emits valid JSON" an assertion rather than a check against ourselves.
fn field_of(written: &str, field: &str) -> serde_json::Value {
    let parsed: serde_json::Value =
        serde_json::from_str(written).expect("the writer emits valid JSON");
    parsed
        .get(field)
        .unwrap_or_else(|| panic!("the written config has no `{field}`"))
        .clone()
}

/// What `slot` becomes after a read and a write.
fn written(slot: &str, json: &str) -> serde_json::Value {
    field_of(&rewrite(&config(&[(slot, json)])), slot)
}

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("a test expectation is valid JSON")
}

fn ids(tokenizer: &PipelineTokenizer, text: &str, specials: bool) -> Vec<u32> {
    tokenizer
        .encode(text, specials)
        .wait()
        .expect("encoding a fixture text")
        .iter()
        .flat_map(|encoding| encoding.ids())
        .map(|token| token.id())
        .collect()
}

/// `data/tokenizer.json` is excluded: it is test *output*, written by another test binary, and cargo
/// runs those in parallel.
fn fixtures() -> Vec<std::path::PathBuf> {
    let dir = std::path::Path::new("../data");
    if !dir.exists() {
        return Vec::new();
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read data/")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .filter(|path| path.file_name().is_some_and(|name| name != "tokenizer.json"))
        .collect();
    files.sort();
    files
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
    "tabs\tand\nnewlines\r\nand  double  spaces",
    "   ",
    "",
];

// ---- the gate --------------------------------------------------------------------------------

/// Read a real config, write it, read that back, encode with both. A reader refusal is a skip; a
/// *writer* refusal is a failure -- everything the reader builds, the writer must describe.
// TODO(pr3): every fixture in `data/` is a legacy 1.0 file, so this has no input until tk-convert
// can produce 2.0 ones. Port it there.
#[ignore = "needs 2.0 fixtures from tk-convert (pr3)"]
#[test]
fn round_trip_preserves_ids_on_every_real_config() {
    let mut configs = 0usize;
    for path in fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read a fixture");
        let Ok(before) = from_json(&text) else {
            continue;
        };
        let written = to_json(&before)
            .unwrap_or_else(|e| panic!("{name}: the reader built it, the writer cannot: {e}"));
        let after = from_json(&written)
            .unwrap_or_else(|e| panic!("{name}: the writer's output its own reader refuses: {e}"));
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

/// Every Unigram score, not a sample: one ULP flips a Viterbi near-tie, and `TEXTS` would not
/// necessarily contain it. The literal must read back *through this crate's parser* to the same
/// bits -- which is not free, because that parser reproduces `serde_json`'s arithmetic rather than
/// being correctly rounded.
// TODO(pr3): needs 2.0 Unigram fixtures, as above.
#[ignore = "needs 2.0 fixtures from tk-convert (pr3)"]
#[test]
#[cfg(all(feature = "unigram", feature = "normalizers"))]
fn every_unigram_score_survives_the_writer_bit_for_bit() {
    let mut models = 0usize;
    for path in fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read a fixture");
        let Ok(tokenizer) = from_json(&text) else {
            continue;
        };
        // From the built pipeline, never the raw text: `albert-base-v1` has no `"type"` at all.
        let PipelineModel::Unigram(unigram) = tokenizer.get_model() else {
            continue;
        };
        models += 1;
        for (token, score) in unigram.vocab() {
            let literal = super::writer::float_literal(*score)
                .unwrap_or_else(|e| panic!("{name}: the score for {token:?} has no spelling: {e}"));
            let read_back = crate::vendored::f64_from_literal(&literal);
            assert_eq!(
                read_back.to_bits(),
                score.to_bits(),
                "{name}: {token:?} was written {literal}, which reads back as {read_back}"
            );
        }
    }
    assert!(models >= 1, "no Unigram config was checked");
}

// ---- the gate has teeth ----------------------------------------------------------------------

/// Merges that *compete*: `a+b` and `b+c` both apply to `abc`, so rank alone picks the winner.
/// [`BPE_MODEL`] cannot serve -- its merges form a chain, and a chain offers no choice, so
/// reversing it is as inert as `merges.pop()`.
const COMPETING_BPE: &str = r#"{"type": "BPE", "byte_level": false,
    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4}, "merges": [["a", "b"], ["b", "c"]]}"#;

/// Perturb the *written config*, not the writer, so this cannot rot: it shows that the comparison
/// above is sensitive to a wrong merge order, whoever produced it.
#[test]
fn reversing_the_written_merges_moves_ids() {
    let tokenizer = from_json(&config(&[("model", COMPETING_BPE)])).expect("it reads");
    let mut parsed: serde_json::Value =
        serde_json::from_str(&to_json(&tokenizer).expect("and writes")).expect("valid JSON");
    let merges = parsed["model"]["merges"].as_array_mut().expect("merges");
    assert_eq!(*merges, json(r#"[["a", "b"], ["b", "c"]]"#), "not rank order");
    merges.reverse();
    let perturbed = from_json(&parsed.to_string()).expect("the perturbed config still reads");

    // `a+b` outranks `b+c`, so `abc` is `ab` + `c`. Reversed, `b+c` wins: `a` + `bc`.
    assert_eq!(ids(&tokenizer, "abc", false), vec![3, 2]);
    assert_eq!(
        ids(&perturbed, "abc", false),
        vec![0, 4],
        "reversing the merge order left the ids alone, so the gate compares nothing"
    );
}

/// The two weaker perturbations, recorded so nobody rediscovers that they are inert: popping the
/// last merge, and reversing a *chain*. Both look exactly like a dead gate.
#[test]
fn the_weak_perturbations_really_are_inert() {
    let tokenizer = from_json(&config(&[])).expect("the tiny config reads");
    let text = to_json(&tokenizer).expect("and writes");
    assert_eq!(ids(&tokenizer, "abab", false), vec![3], "`abab` merges up");

    for pop in [true, false] {
        let mut parsed: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let merges = parsed["model"]["merges"].as_array_mut().expect("merges");
        if pop {
            merges.pop();
        } else {
            merges.reverse();
        }
        let perturbed = from_json(&parsed.to_string())
            .unwrap_or_else(|_| panic!("pop={pop} was refused, which would make it a fine gate"));
        assert_eq!(
            ids(&perturbed, "abab", false),
            vec![3],
            "pop={pop} moved an id after all, so it should become a real perturbation"
        );
    }
}

/// The same perturbation against the fixtures, so the gate is shown to bite on what it guards.
/// Being *refused* counts: a byte-level model needs every byte to be an atom.
// TODO(pr3): needs 2.0 fixtures, as above.
#[ignore = "needs 2.0 fixtures from tk-convert (pr3)"]
#[test]
fn reversing_the_written_merges_moves_ids_on_a_real_config() {
    let mut bitten = 0usize;
    for path in fixtures() {
        let text = std::fs::read_to_string(&path).expect("read a fixture");
        let Ok(before) = from_json(&text) else {
            continue;
        };
        let Ok(written) = to_json(&before) else {
            continue;
        };
        let mut parsed: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");
        let Some(merges) = parsed["model"]["merges"].as_array_mut() else {
            continue;
        };
        if merges.len() < 2 {
            continue;
        }
        merges.reverse();
        match from_json(&parsed.to_string()) {
            Err(_) => bitten += 1,
            Ok(after) => {
                if TEXTS
                    .iter()
                    .any(|t| ids(&before, t, false) != ids(&after, t, false))
                {
                    bitten += 1;
                }
            }
        }
    }
    assert!(
        bitten >= 1,
        "reversing every BPE fixture's merges changed nothing, so the comparison cannot see a \
         merge-order defect at all"
    );
}

// ---- canonical spelling, one row per component -----------------------------------------------

/// `(what goes in, what must come out)`. A nested `Sequence` flattens and an empty one disappears.
/// `drop_whitespace` is one flag, never the legacy `Sequence[WhitespaceSplit, Metaspace]`.
#[rustfmt::skip]
const NORMALIZERS: &[(&str, &str)] = &[
    (r#"{"type": "Lowercase"}"#, r#"{"type": "Lowercase"}"#),
    (r#"{"type": "Sequence", "normalizers": []}"#, "null"),
    (r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"},
        {"type": "Sequence", "normalizers": [{"type": "Strip", "strip_left": true, "strip_right": false}]},
        {"type": "Prepend", "prepend": "_"}]}"#,
     r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"},
        {"type": "Strip", "strip_left": true, "strip_right": false}, {"type": "Prepend", "prepend": "_"}]}"#),
    (r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#,
     r#"{"type": "Replace", "pattern": {"String": " "}, "content": "_"}"#),
    (r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": true, "drop_whitespace": false}"#,
     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": true, "drop_whitespace": false}"#),
    (r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": true, "drop_whitespace": true}"#,
     r#"{"type": "MetaspaceNormalizer", "replacement": "▁", "prepend": true, "drop_whitespace": true}"#),
];

#[rustfmt::skip]
const PRE_TOKENIZERS: &[(&str, &str)] = &[
    (r#"{"type": "Digits", "individual_digits": true}"#, r#"{"type": "Digits", "individual_digits": true}"#),
    (r#"{"type": "Whitespace"}"#, r#"{"type": "Whitespace"}"#),
    (r#"{"type": "WhitespaceSplit"}"#, r#"{"type": "WhitespaceSplit"}"#),
    (r#"{"type": "BertPreTokenizer"}"#, r#"{"type": "BertPreTokenizer"}"#),
    (r#"{"type": "Punctuation", "behavior": "Removed"}"#, r#"{"type": "Punctuation", "behavior": "Removed"}"#),
    (r#"{"type": "CharDelimiterSplit", "delimiter": "-"}"#, r#"{"type": "CharDelimiterSplit", "delimiter": "-"}"#),
    (r#"{"type": "FixedLength", "length": 7}"#, r#"{"type": "FixedLength", "length": 7}"#),
    (r#"{"type": "Split", "pattern": {"String": "-"}, "behavior": "MergedWithNext", "invert": false}"#,
     r#"{"type": "Split", "pattern": {"String": "-"}, "behavior": "MergedWithNext", "invert": false}"#),
    (r#"{"type": "Sequence", "pretokenizers": [{"type": "Whitespace"}, {"type": "Digits", "individual_digits": false}]}"#,
     r#"{"type": "Sequence", "pretokenizers": [{"type": "Whitespace"}, {"type": "Digits", "individual_digits": false}]}"#),
];

/// A decoder keeps all three prepend schemes -- nothing here has to be expressible as a normalizer,
/// so `first` survives. `ByteLevel`'s flags and `Metaspace`'s `split` are read past, not carried:
/// decoding is a fixed inverse of the byte map and never looks at either.
#[rustfmt::skip]
const DECODERS: &[(&str, &str)] = &[
    (r#"{"type": "Fuse"}"#, r#"{"type": "Fuse"}"#),
    (r#"{"type": "ByteFallback"}"#, r#"{"type": "ByteFallback"}"#),
    (r#"{"type": "Strip", "content": "_", "start": 1, "stop": 0}"#,
     r#"{"type": "Strip", "content": "_", "start": 1, "stop": 0}"#),
    (r#"{"type": "BPEDecoder", "suffix": "</w>"}"#, r#"{"type": "BPEDecoder", "suffix": "</w>"}"#),
    (r###"{"type": "WordPiece", "prefix": "##", "cleanup": true}"###,
     r###"{"type": "WordPiece", "prefix": "##", "cleanup": true}"###),
    (r#"{"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}"#,
     r#"{"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}"#),
    (r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
     r#"{"type": "ByteLevel"}"#),
    (r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}"#,
     r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}"#),
    (r#"{"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}"#,
     r#"{"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}"#),
];

/// Every post-processor is two templates of sequence markers and runs of ids once it reaches the
/// pipeline, so all three spellings come back as one: pieces carrying their own ids, `type_id` only
/// where it is not 0. Roberta's doubled separator is one piece standing for two ids.
#[rustfmt::skip]
const POST_PROCESSORS: &[(&str, &str)] = &[
    (r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}],
        "pair": [{"seq": "A"}, {"seq": "B", "type_id": 1}]}"#,
     r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}],
        "pair": [{"seq": "A"}, {"seq": "B", "type_id": 1}]}"#),
    (r#"{"type": "BertProcessing", "cls": ["a", 0], "sep": ["b", 1]}"#,
     r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}, {"ids": [1]}],
        "pair": [{"ids": [0]}, {"seq": "A"}, {"ids": [1]}, {"seq": "B", "type_id": 1}, {"ids": [1], "type_id": 1}]}"#),
    (r#"{"type": "RobertaProcessing", "cls": ["a", 0], "sep": ["b", 1], "trim_offsets": true, "add_prefix_space": true}"#,
     r#"{"type": "TemplateProcessing", "single": [{"ids": [0]}, {"seq": "A"}, {"ids": [1]}],
        "pair": [{"ids": [0]}, {"seq": "A"}, {"ids": [1, 1]}, {"seq": "B"}, {"ids": [1]}]}"#),
];

/// `strip_accents: null` means "decide from `lowercase`", which is why the reader requires the key.
#[cfg(feature = "normalizers")]
#[rustfmt::skip]
const BERT_NORMALIZER: &[(&str, &str)] = &[
    (r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}"#,
     r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}"#),
];

#[test]
fn components_round_trip_to_their_canonical_spelling() {
    let mut slots: Vec<(&str, &[(&str, &str)])> = vec![
        ("normalizer", NORMALIZERS),
        ("pre_tokenizer", PRE_TOKENIZERS),
        ("decoder", DECODERS),
        ("post_processor", POST_PROCESSORS),
    ];
    #[cfg(feature = "normalizers")]
    slots.push(("normalizer", BERT_NORMALIZER));

    for (slot, cases) in slots {
        for (input, expected) in cases {
            assert_eq!(written(slot, input), json(expected), "{slot}: {input}");
        }
    }
}

#[test]
fn the_canonical_shape_is_tagged_versioned_and_null_where_absent() {
    let text = rewrite(&config(&[
        ("normalizer", r#"{"type": "Lowercase"}"#),
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
    assert_eq!(parsed["model"]["merges"], json(r#"[["a","b"],["ab","ab"]]"#));

    // A pass-through frame is what "no post-processor" lowers to, so it goes back out as absent.
    let bare = rewrite(&config(&[]));
    for field in ["normalizer", "pre_tokenizer", "decoder", "post_processor"] {
        assert_eq!(field_of(&bare, field), serde_json::Value::Null, "{field}");
    }
    assert_eq!(field_of(&bare, "truncation"), serde_json::Value::Null);
    assert_eq!(field_of(&bare, "padding"), serde_json::Value::Null);
}

/// A `MetaspaceNormalizer` is a normalizer like any other now, so its position carries no meaning
/// and the writer never has to fold it back into a pre-tokenizer tag -- nor write its legacy keys.
#[test]
fn a_metaspace_normalizer_writes_wherever_it_sits() {
    use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
    use tk_encode::normalizers::utils::Lowercase;

    let chain = vec![
        PipelineNormalizer::Metaspace(MetaspaceNormalizer::new('\u{2581}', true, false)),
        PipelineNormalizer::Lowercase(Lowercase),
    ];
    let mut out = super::writer::Out::new();
    super::normalizers::write_normalizer(&mut out, &chain).expect("a Metaspace anywhere writes");
    let parsed: serde_json::Value = serde_json::from_str(&out.finish()).expect("valid JSON");
    assert_eq!(parsed["normalizers"][0]["type"], "MetaspaceNormalizer");
    assert_eq!(parsed["normalizers"][1]["type"], "Lowercase");
    for legacy in ["add_prefix_space", "prepend_scheme", "split"] {
        assert!(
            parsed["normalizers"][0].get(legacy).is_none(),
            "wrote `{legacy}`"
        );
    }
}

/// Ascending id order is load-bearing: the reader replays added tokens in that order, and
/// `add_tokens` reuses a model id when the token is already in the vocabulary.
#[test]
#[rustfmt::skip]
fn added_tokens_go_out_in_id_order_with_every_flag() {
    let added = r#"[
        {"id": 5, "content": "<b>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true},
        {"id": 4, "content": "<a>", "single_word": true, "lstrip": false, "rstrip": true, "normalized": true, "special": false}]"#;
    let expected = r#"[
        {"id": 4, "content": "<a>", "single_word": true, "lstrip": false, "rstrip": true, "normalized": true, "special": false},
        {"id": 5, "content": "<b>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true}]"#;
    assert_eq!(written("added_tokens", added), json(expected));
}

/// Asserted on *bits*, not text: the writer emits the shortest spelling of the double **our parser
/// produced**, which for `-3.8403830528259277` is a different double from the correctly-rounded
/// reading of those digits -- so its shortest form has sixteen digits where the file had seventeen.
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
    for (entry, digits) in vocab
        .iter()
        .zip(["0.0", "-3.8403830528259277", "-13.5321998596191"])
    {
        let literal = entry[1].to_string();
        let got = crate::vendored::f64_from_literal(&literal);
        let want = crate::vendored::f64_from_literal(digits);
        assert_eq!(got.to_bits(), want.to_bits(), "{literal} reads back as {got}");
    }
    // And a score is spelled as a float, never as a bare integer.
    assert_eq!(vocab[0][1].to_string(), "0.0");
}

#[cfg(feature = "wordpiece")]
#[test]
#[rustfmt::skip]
fn a_wordpiece_model_keeps_all_four_required_fields() {
    let model = r###"{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1, "##b": 2}}"###;
    assert_eq!(written("model", model), json(model));
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
