//! Pre-tokenizer lowering: `PreTokenizerWrapper` -> `PipelinePreTokenizer`.
//!
//! Three groups, all moved here with the lowering they exercise: the `Sequence` differential
//! against the legacy path (including deepseek's exact 3-`Split` composition, which routes to an
//! `atomsplit` FSM), the `ByteLevel` rewrite into a `Split` on the GPT-2 pattern, and the
//! `Metaspace` decomposition into a normalizer plus a `Split`.

use std::convert::TryInto;

use tk_convert::PreTokenizerWrapper;
use tk_convert::lowering::to_normalizer_and_split;
use tk_convert::pre_tokenizers::Sequence;
use tk_encode::pipeline::{self, PipelinePreTokenizer};
use tk_encode::pre_tokenizers::byte_level::ByteLevel;
use tk_encode::pre_tokenizers::sequence::PipelineSequence;
use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;
use tk_encode::{PreTokenizedString, PreTokenizer};

use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::whitespace::Whitespace;
use tk_encode::pre_tokenizers::{punctuation::Punctuation, whitespace::WhitespaceSplit};
use tk_encode::{OffsetReferential, OffsetType};

/// Run the pipeline path and return `(piece, (start, end))` for each split.
fn pipeline_pretokenize(seq: &PipelineSequence, text: &str) -> Vec<(String, (usize, usize))> {
    let mut scratch = pipeline::PreTokenizerScratch::default();
    let mut out = Vec::new();
    tk_encode::pipeline::PreTokenizer::pre_tokenize(seq, text, &mut scratch, &mut out).unwrap();
    out.iter()
        .map(|s| {
            (
                text[s.range()].to_string(),
                (s.start as usize, s.end as usize),
            )
        })
        .collect()
}

/// The legacy path's `(piece, offsets)` — the oracle.
fn legacy_pretokenize(seq: &Sequence, text: &str) -> Vec<(String, (usize, usize))> {
    let mut pre = PreTokenizedString::from(text);
    PreTokenizer::pre_tokenize(seq, &mut pre).unwrap();
    pre.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (s.to_string(), o))
        .collect()
}

#[test]
fn pipeline_sequence_basic() {
    // Same config + expectation as `sequence_basic`, via the range API.
    let seq = Sequence::new(vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Punctuation(Punctuation::default()),
    ]);
    let pipe_seq = seq
        .clone()
        .try_into()
        .expect("Failed to convert Sequence to PipelineSequence");
    assert_eq!(
        pipeline_pretokenize(&pipe_seq, "Hey friend!     How are you?!?"),
        [
            ("Hey", (0, 3)),
            ("friend", (4, 10)),
            ("!", (10, 11)),
            ("How", (16, 19)),
            ("are", (20, 23)),
            ("you", (24, 27)),
            ("?", (27, 28)),
            ("!", (28, 29)),
            ("?", (29, 30)),
        ]
        .map(|(s, o)| (s.to_string(), o)),
    );
}

#[test]
fn pipeline_matches_legacy_oracle() {
    // Differential: the pipeline path must equal the legacy path across
    // varied configs (incl. a nested Sequence) and multi-script texts.
    let configs: Vec<Vec<PreTokenizerWrapper>> = vec![
        vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ],
        vec![PreTokenizerWrapper::Whitespace(Whitespace)],
        vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Digits(Digits::new(true)),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ],
        // nested Sequence as a child
        vec![
            PreTokenizerWrapper::Sequence(Sequence::new(vec![
                PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            ])),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ],
    ];
    let texts = [
        "Hey friend!     How are you?!?",
        "abc 123 def!!ghi 42",
        "  leading  and   trailing spaces  ",
        "café? no—maybe 3.14 ok",
        "中文 text 123, mixed!",
        "single",
        "!!!",
    ];
    for (ci, cfg) in configs.into_iter().enumerate() {
        let seq = Sequence::new(cfg);
        let pipe_seq = seq
            .clone()
            .try_into()
            .expect("Failed to convert Sequence to PipelineSequence");
        for text in texts {
            assert_eq!(
                pipeline_pretokenize(&pipe_seq, text),
                legacy_pretokenize(&seq, text),
                "config #{ci} diverged on {text:?}",
            );
        }
    }
}

#[cfg(feature = "fancy-regex")] // deepseek `Split`s need a backend at construction (legacy baseline)
#[test]
fn pipeline_deepseek_uses_fsm_and_matches_legacy() {
    // Load deepseek-v4's real pre_tokenizer, rebuild a Sequence of just its 3 Splits (drop the
    // trailing byte-map ByteLevel), and prove: (1) the exact fixture patterns are recognized,
    // (2) the fsm_deepseek pipeline output == the 3-regex-split legacy output, byte-for-byte.
    let path = "../data/deepseek-v4-flash-base-tokenizer.json";
    if !std::path::Path::new(path).exists() {
        return; // fixture not downloaded in this environment
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let splits: Vec<PreTokenizerWrapper> = v["pre_tokenizer"]["pretokenizers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["type"] == "Split")
        .map(|c| serde_json::from_value(c.clone()).unwrap())
        .collect();
    assert_eq!(splits.len(), 3, "deepseek has 3 Splits");
    let seq = Sequence::new(splits);
    let pipe: PipelineSequence = seq.clone().try_into().unwrap();
    assert!(
        pipe.is_deepseek(),
        "deepseek's exact 3-Split sequence must be recognized"
    );

    for text in [
        "中文 with 123 numbers!! and ケーキ don't",
        "hello 世界\n\n表 x",
        "純粋なCJK日本語テキスト",
        "  spaces  and\ttabs 42 café Naïve",
    ] {
        assert_eq!(
            pipeline_pretokenize(&pipe, text),
            legacy_pretokenize(&seq, text),
            "deepseek diverged on {text:?}",
        );
    }
}

// CJK-range PUNCTUATION (・ U+30FB, ゠, ゛゜) sits inside Split-1's `[一-龥぀-ゟ゠-ヿ]` range, so
// Split-1 isolates it (`fsm_deepseek` handles a CJK-range run as a closed unit) — a preceding space
// stays separate and it never merges with adjacent non-CJK punct.
#[cfg(feature = "fancy-regex")] // deepseek `Split`s need a backend at construction (legacy baseline)
#[test]
fn pipeline_deepseek_cjk_punct_whitespace_edge() {
    let path = "../data/deepseek-v4-flash-base-tokenizer.json";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let splits: Vec<PreTokenizerWrapper> = v["pre_tokenizer"]["pretokenizers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["type"] == "Split")
        .map(|c| serde_json::from_value(c.clone()).unwrap())
        .collect();
    let seq = Sequence::new(splits);
    let pipe: PipelineSequence = seq.clone().try_into().unwrap();
    let text = "hello 世界\n\n表 ・ x"; // standalone ・ with surrounding spaces
    assert_eq!(
        pipeline_pretokenize(&pipe, text),
        legacy_pretokenize(&seq, text)
    );
}

// fsm_deepseek == the 3-Split onig Sequence over multilingual Wikipedia corpora — the broad byte-exact
// guard. `he.txt` is why it exists: Hebrew mixes format controls (RLM, `\p{Cf}`) and Other_Alphabetic
// symbols (Ⓘ, `\p{S}` but is_alphabetic), which stress the *gap* grouping (consecutive unmatched chars
// = one piece) and the `ALPHA_SYM` Mark refinement (a `\w` char that is NOT `[\p{L}\p{M}]`).
#[cfg(feature = "fancy-regex")] // deepseek `Split`s need a backend at construction (legacy baseline)
#[test]
fn pipeline_deepseek_matches_legacy_on_corpora() {
    let path = "../data/deepseek-v4-flash-base-tokenizer.json";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let splits: Vec<PreTokenizerWrapper> = v["pre_tokenizer"]["pretokenizers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["type"] == "Split")
        .map(|c| serde_json::from_value(c.clone()).unwrap())
        .collect();
    let seq = Sequence::new(splits);
    let pipe: PipelineSequence = seq.clone().try_into().unwrap();
    assert!(pipe.is_deepseek());
    // `he`/`ar` (RTL, RLM/format-mark + Other_Alphabetic-symbol heavy) are the cases the atomsplit
    // deepseek bench doesn't cover; the other 8 languages are byte-exact-gated there.
    for lang in ["he", "ar"] {
        let Ok(corpus) = std::fs::read_to_string(format!("../atomsplit/benches/data/{lang}.txt"))
        else {
            continue;
        };
        for (ln, line) in corpus.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (p, l) = (
                pipeline_pretokenize(&pipe, line),
                legacy_pretokenize(&seq, line),
            );
            if p != l {
                let k = p
                    .iter()
                    .zip(l.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(p.len().min(l.len()));
                let lo = k.saturating_sub(1);
                panic!(
                    "deepseek diverged {lang}.txt:{ln} @tok {k}\n  {line:?}\n  pipe: {:?}\n  legc: {:?}",
                    &p[lo..(k + 3).min(p.len())],
                    &l[lo..(k + 3).min(l.len())],
                );
            }
        }
    }
}

#[test]
fn pipeline_empty_input() {
    let seq = Sequence::new(vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Punctuation(Punctuation::default()),
    ]);
    let pipe_seq = seq
        .clone()
        .try_into()
        .expect("Failed to convert Sequence to PipelineSequence");

    assert!(pipeline_pretokenize(&pipe_seq, "").is_empty());
}

#[test]
fn pipeline_matches_legacy_oracle_byte_level() {
    // The Llama-3 / DeepSeek archetype: Sequence[Split(regex), ByteLevel(use_regex=false)].
    // Pipeline ranges must match the legacy oracle's Original-referential offsets, and the
    // byte-level transform of each range must match the legacy split string.
    use tk_encode::SplitDelimiterBehavior;
    use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
    use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;
    use tk_encode::utils::byte_level::GPT2_REGEX_STR;

    let seq = Sequence::new(vec![
        PreTokenizerWrapper::Split(
            Split::new(
                SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
    ]);
    let pipe_seq: PipelineSequence = seq
        .clone()
        .try_into()
        .expect("Failed to convert Sequence to PipelineSequence");
    for text in [
        "Hello there\nHello there",
        "中文 text 123, mixed! 🤗",
        "I'm sure it's fine   ",
    ] {
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut out = Vec::new();
        tk_encode::pipeline::PreTokenizer::pre_tokenize(&pipe_seq, text, &mut scratch, &mut out)
            .unwrap();
        let pipeline: Vec<(String, (usize, usize))> = out
            .iter()
            .map(|s| {
                let transformed = text[s.range()]
                    .bytes()
                    .map(|b| BYTES_CHAR_LOOKUP[b as usize])
                    .collect();
                (transformed, (s.start as usize, s.end as usize))
            })
            .collect();
        assert_eq!(
            pipeline,
            legacy_pretokenize(&seq, text),
            "diverged on {text:?}"
        );
    }
}

#[test]
fn deserialized_sequence_matches_legacy_oracle() {
    // Real tokenizers are loaded via serde, not `Sequence::new` — the pipeline
    // path must behave identically for a deserialized Sequence.
    let seq: Sequence =
        serde_json::from_str(r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"}]}"#)
            .unwrap();
    let pipe_seq = seq
        .clone()
        .try_into()
        .expect("Failed to convert Sequence to PipelineSequence");

    let text = "Hey friend!     How are you?!?";
    assert_eq!(
        pipeline_pretokenize(&pipe_seq, text),
        legacy_pretokenize(&seq, text),
    );
}

#[test]
fn pipeline_unsupported_child_errors() {
    // Metaspace has no range-based form. Constructing the Sequence must still work
    // (the legacy path supports it) — only the pipeline conversion should fail.
    let seq = Sequence::new(vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Metaspace(tk_encode::pre_tokenizers::metaspace::Metaspace::default()),
    ]);
    assert!(PipelinePreTokenizer::try_from(PreTokenizerWrapper::Sequence(seq)).is_err());
}

#[test]
fn sequence_basic() {
    let pretokenizers = vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Punctuation(Punctuation::default()),
    ];
    let pretok = Sequence::new(pretokenizers);
    let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
    pretok.pre_tokenize(&mut pretokenized).unwrap();
    assert_eq!(
        pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s, o))
            .collect::<Vec<_>>(),
        vec![
            ("Hey", (0, 3)),
            ("friend", (4, 10)),
            ("!", (10, 11)),
            ("How", (16, 19)),
            ("are", (20, 23)),
            ("you", (24, 27)),
            ("?", (27, 28)),
            ("!", (28, 29)),
            ("?", (29, 30)),
        ]
    );
}

/// Splits from the pipeline conversion of `byte_level`, with the raw text of each
/// range transformed to the byte-level alphabet so it's comparable with the legacy
/// oracle's output strings.
fn pipeline_splits(byte_level: ByteLevel, text: &str) -> Vec<(String, (usize, usize))> {
    let converted = <PipelinePreTokenizer as std::convert::TryFrom<_>>::try_from(
        tk_convert::PreTokenizerWrapper::ByteLevel(byte_level),
    )
    .unwrap();
    let mut scratch = tk_encode::pipeline::PreTokenizerScratch::default();
    let mut out = Vec::new();
    tk_encode::pipeline::PreTokenizer::pre_tokenize(&converted, text, &mut scratch, &mut out)
        .unwrap();
    out.iter()
        .map(|s| {
            let transformed = text[s.range()]
                .bytes()
                .map(|b| BYTES_CHAR_LOOKUP[b as usize])
                .collect();
            (transformed, (s.start as usize, s.end as usize))
        })
        .collect()
}

fn legacy_splits(byte_level: ByteLevel, text: &str) -> Vec<(String, (usize, usize))> {
    let mut pre = PreTokenizedString::from(text);
    byte_level.pre_tokenize(&mut pre).unwrap();
    pre.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, o, _)| (s.to_string(), o))
        .collect()
}

#[test]
fn pipeline_conversion_matches_legacy_splits() {
    let byte_level = ByteLevel::default().add_prefix_space(false);
    for text in [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there       dear",
        " leading space",
        "trailing space   ",
        "i⭢j",
        "中文 text 123, mixed! 🤗 emoji",
        "I'm can't we've they'll it's",
        "tabs\tand\r\nnewlines",
        "café über naïve",
        "!!!???...",
        "single",
    ] {
        assert_eq!(
            pipeline_splits(byte_level, text),
            legacy_splits(byte_level, text),
            "diverged on {text:?}",
        );
    }
}

#[test]
fn pipeline_conversion_no_regex_is_identity_split() {
    let byte_level = ByteLevel::default()
        .add_prefix_space(false)
        .use_regex(false);
    let text = "Hello my friend, how is your day going?";
    assert_eq!(
        pipeline_splits(byte_level, text),
        legacy_splits(byte_level, text),
    );
}

#[test]
fn pipeline_conversion_rejects_add_prefix_space() {
    // The range-based pipeline can't prepend text; converting must fail loudly
    // rather than silently produce different splits than the legacy path.
    let byte_level = ByteLevel::default().add_prefix_space(true);
    assert!(
        <PipelinePreTokenizer as std::convert::TryFrom<_>>::try_from(
            tk_convert::PreTokenizerWrapper::ByteLevel(byte_level)
        )
        .is_err()
    );
}

fn pre_tokenizer_from(json: &str) -> PreTokenizerWrapper {
    serde_json::from_str(json).unwrap()
}

/// t5 and albert: throw the whitespace away, then start every word with `▁`.
const T5: &str = r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
/// A `Metaspace` on its own: each space becomes `▁`, tabs and newlines stay.
const BARE: &str =
    r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#;
/// Nothing ties the delimiter to `▁`, and a 1-byte one takes a different code path in
/// `Split`, so keep an ASCII delimiter under test too.
const ASCII_DELIMITER: &str =
    r#"{"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true}"#;

/// Every kind of gap, plus text that already holds the delimiter.
const TEXTS: &[&str] = &[
    "hello world",
    "hello   world",
    " leading",
    "trailing ",
    "  both  ",
    "one\ttab\nand a newline",
    "\tleading tab",
    // A gap that is whitespace to `char::is_whitespace` but not to an ASCII scan:
    // a no-break space and an ideographic space.
    "nbsp\u{a0}gap and\u{3000}an ideographic space",
    "▁already marked",
    "a▁b c",
    "▁▁▁a b",
    "▁",
    "_underscored_ text",
    "single",
    "   ",
    "",
];

/// Normalizing and then splitting must produce exactly the words the `Metaspace`
/// pre-tokenizer produces on its own: it is the behaviour being rebuilt, so it is the
/// reference.
fn assert_words_match_the_pre_tokenizer(json: &str) {
    let declared = pre_tokenizer_from(json);
    let (normalizer, split) =
        to_normalizer_and_split(Some(&declared)).expect("this shape is supported");
    for text in TEXTS {
        let mut legacy = PreTokenizedString::from(*text);
        declared.pre_tokenize(&mut legacy).unwrap();
        let expected: Vec<&str> = legacy
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .iter()
            .map(|(word, _, _)| *word)
            .collect();

        let normalized = pipeline::Normalizer::normalize(&normalizer, text).unwrap();
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut spans = Vec::new();
        pipeline::PreTokenizer::pre_tokenize(&split, &normalized, &mut scratch, &mut spans)
            .unwrap();
        let words: Vec<&str> = spans.iter().map(|s| &normalized[s.range()]).collect();
        assert_eq!(words, expected, "{text:?}");
    }
}

#[test]
fn t5_shape_matches_its_pre_tokenizer() {
    assert_words_match_the_pre_tokenizer(T5);
}

#[test]
fn bare_metaspace_matches_its_pre_tokenizer() {
    assert_words_match_the_pre_tokenizer(BARE);
}

#[test]
fn ascii_delimiter_matches_its_pre_tokenizer() {
    assert_words_match_the_pre_tokenizer(ASCII_DELIMITER);
}

#[test]
fn refuses_what_it_cannot_reproduce() {
    let refused = [
        // Delimiters written but the text never cut: no `Split` step to hand back.
        (
            "a metaspace that does not split",
            r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":false}"#,
        ),
        // `first` marks a piece only when it opened the text it came from; a normalizer is
        // given chunks, not their position.
        (
            "a metaspace that prepends to the first word only",
            r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}"#,
        ),
        // Whitespace gone and no delimiter written: nothing left to show where words start.
        (
            "dropped whitespace and no prepending",
            r#"{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}]}"#,
        ),
        // Not a metaspace shape at all.
        ("a bare whitespace split", r#"{"type":"WhitespaceSplit"}"#),
    ];
    for (name, json) in refused {
        assert!(
            to_normalizer_and_split(Some(&pre_tokenizer_from(json))).is_none(),
            "{name}"
        );
    }
    assert!(to_normalizer_and_split(None).is_none(), "no pre-tokenizer");
}

/// The real files, so a config shape drifting out of the two above shows up here instead of
/// silently skipping the pipeline oracle for these models. Skipped when they are not fetched.
///
/// Both configs declare table-backed normalizers (t5 a `Precompiled` charsmap, albert an
/// `NFKD`/`StripAccents` sequence) *and* both are untagged Unigram models (`model` carries
/// `unk_id` plus a `[token, score]` vocab list and no `"type"`) — every one of those shapes is
/// this crate's job to read.
#[test]
fn real_configs_convert() {
    for file in ["t5-base.json", "albert-base-v1-tokenizer.json"] {
        let path = format!("../data/{file}");
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let tokenizer = tk_convert::Tokenizer::from_file(&path).unwrap();
        assert!(
            to_normalizer_and_split(tokenizer.get_pre_tokenizer()).is_some(),
            "{file} should convert"
        );
    }
}
