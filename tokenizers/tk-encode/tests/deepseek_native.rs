//! deepseek's pre-tokenizer is a `Sequence` of three `Split`s whose regexes `atomsplit` reproduces
//! natively, so a deepseek `tokenizer.json` must **load and encode with no system-regex backend** —
//! `fancy-regex` off, which is the default build. Recognition happens per-`Split` (that is what lets
//! `Split::new` accept a missing backend), and the `Sequence` then fuses the recognized chain into one
//! `fsm_deepseek` pass.
//!
//! The pipeline (fused) and legacy (three chained splits) paths must agree on ids. That comparison is
//! meaningful in BOTH builds, and checks a different thing in each: with `fancy-regex` the legacy side
//! is oniguruma — the reference; without it, the legacy side is the three standalone FSMs, so the test
//! pins fused == chained end-to-end (`atomsplit`'s `parity.rs` gates both against oniguruma).

mod common;

use std::convert::TryFrom;
use std::path::Path;

use common::{DATA, FIXTURES, WINDOWS, window};
use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const MODEL: &str = "deepseek-v4.json";

const PROBE: &str = "中文 with 123 numbers!! and ケーキ don't\n\n  純粋なCJK日本語テキスト x";

fn load() -> Option<(Tokenizer, PipelineTokenizer)> {
    let path = Path::new(DATA).join(MODEL);
    if !path.exists() {
        eprintln!("skip: {MODEL} absent (fetch with `make bench-models`)");
        return None;
    }
    // The load itself is the assertion: with no backend compiled, an unrecognized `Split` regex is a
    // hard error here, so this only succeeds because all three deepseek patterns are recognized.
    let tree =
        Tokenizer::from_file(&path).expect("deepseek must load with no system-regex backend");
    let pipeline = PipelineTokenizer::try_from(&tree).expect("pipeline must build");
    Some((tree, pipeline))
}

fn ids(tree: &Tokenizer, pipeline: &PipelineTokenizer, text: &str) -> (Vec<u32>, Vec<u32>) {
    let legacy = tree
        .encode_fast(text, false)
        .expect("legacy encode")
        .get_ids()
        .to_vec();
    let fused = pipeline
        .encode(text, false)
        .expect("pipeline encode")
        .iter()
        .map(|t| t.id)
        .collect();
    (legacy, fused)
}

#[test]
fn deepseek_encodes_without_a_regex_backend() {
    let Some((tree, pipeline)) = load() else {
        return;
    };
    let (legacy, fused) = ids(&tree, &pipeline, PROBE);
    assert!(!fused.is_empty(), "encode produced no tokens");
    assert_eq!(fused, legacy);
}

#[test]
fn deepseek_fused_matches_chained_splits_on_fixtures() {
    let Some((tree, pipeline)) = load() else {
        return;
    };
    let mut checked = 0;
    for &(group, stem) in FIXTURES {
        let fixture = Path::new(DATA)
            .join("fixtures")
            .join(group)
            .join(format!("{stem}.txt"));
        let Ok(text) = std::fs::read_to_string(&fixture) else {
            continue; // fixture not fetched (`make fixtures`)
        };
        let mut start = 0;
        for &w in WINDOWS {
            let chunk = window(&text, start, w);
            start += w;
            if chunk.is_empty() {
                continue;
            }
            let (legacy, fused) = ids(&tree, &pipeline, chunk);
            assert_eq!(fused, legacy, "diverged on {group}/{stem} window {w}");
            checked += 1;
        }
    }
    eprintln!("windows checked: {checked}");
}
