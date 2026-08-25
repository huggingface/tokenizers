//! The proven fold and the batched (`tokenize_spans`) path must not move a single id.
//!
//! The reference is the released `tokenizers` 0.23.1 encoding of the same strings, captured as
//! literals: the legacy in-crate engine these gates used to compare against is gone, and a gate
//! that compares the pipeline to itself would bless its own regressions. It lives here rather
//! than in `tk-encode` because building a `PipelineTokenizer` from `gpt2.json` needs this crate.
//!
//! Run `make data/gpt2.json` first -- the fixture is fetched, not committed.

use tk_encode::pipeline::PipelineTokenizer;

/// `data/gpt2.json` is still version `1.0`, so run the upgrade pass first -- the reader only
/// accepts canonical `2.0`. Same as `benches/encode.rs`.
fn load() -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file_compact("../data/gpt2.json").unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

/// `<|endoftext|>` is the entry that must NOT fold: it decomposes to seven tokens, and a fold
/// would emit one. The rest mix words that are a single vocabulary entry (folded) with words
/// that are not (merged). gpt2 does not declare `ignore_merges`, so the fold here is on purely
/// because the proof enabled it -- which makes it the config where a wrong proof shows up.
const CASES: &[(&str, &[u32])] = &[
    (
        " the quick brown fox jumps over the lazy dog",
        &[262, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290],
    ),
    (
        "unprefixed words and internationalisation",
        &[403, 3866, 34021, 2456, 290, 3230, 5612],
    ),
    (
        "def foo(bar):\n    return bar + 1\n",
        &[
            4299, 22944, 7, 5657, 2599, 198, 220, 220, 220, 1441, 2318, 1343, 352, 198,
        ],
    ),
    (
        "<|endoftext|> literal in the middle <|endoftext|>",
        &[50256, 18875, 287, 262, 3504, 220, 50256],
    ),
    (
        " 语言模型 mixed with ASCII and ελληνικά",
        &[
            5525, 107, 255, 164, 101, 222, 162, 101, 94, 161, 252, 233, 7668, 351, 37101, 290,
            7377, 113, 39377, 39377, 138, 115, 26180, 29945, 43000, 138, 105,
        ],
    ),
    (
        "aaaaaaaaaaaaaaaaaaaaaaaa",
        &[24794, 24794, 24794, 24794, 24794, 24794],
    ),
    ("   ", &[220, 220, 220]),
    ("", &[]),
];

fn ids(pipe: &PipelineTokenizer, text: &str) -> Vec<u32> {
    pipe.encode(text, false)
        .wait()
        .unwrap()
        .remove(0)
        .ids()
        .iter()
        .map(|t| t.id())
        .collect()
}

#[test]
fn the_proven_fold_never_changes_the_ids() {
    let pipe = load();
    for (text, want) in CASES {
        assert_eq!(
            &ids(&pipe, text)[..],
            *want,
            "the fold changed the ids for {text:?}"
        );
    }
}

#[test]
fn the_batched_path_matches_the_reference() {
    let pipe = load();

    let mut text = String::new();
    for i in 0..400 {
        text.push_str(" the quick brown fox jumps over the lazy dog");
        text.push_str(" internationalisation unfortunately");
        text.push_str(" def foo(bar): return bar + 1");
        text.push_str(" <|xs0|> <|xs1|> <|endoftext|>");
        text.push_str(" 语言模型 ελληνικά");
        if i % 3 == 0 {
            text.push_str(" aaaaaaaaaaaaaaaaaaaaaaaa ");
        }
    }

    // Long enough that spelling every id out would drown the file, so the gate is the count plus
    // an order-sensitive digest -- a reordered or substituted id changes it.
    let got = ids(&pipe, text.as_str());
    assert_eq!(
        got.len(),
        24272,
        "token count differs from the 0.23.1 reference"
    );
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in &got {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    assert_eq!(h, 0x7b5db96b1bf51d01, "the batched path changed the ids");
}
