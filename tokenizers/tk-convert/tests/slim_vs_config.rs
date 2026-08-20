//! The two comparisons between `tk-encode`'s slim JSON reader and this crate's serde reader that
//! are not covered by `examples/json_oracle` (which compares encode over the whole fixture set).
//!
//! Both build the pipeline both ways from the *same* file and demand identical output: the first
//! pins the byte-level merge fold's ids against the reference `Tokenizer`, the second pins `decode`,
//! which `json_oracle` never exercises.

use std::convert::TryFrom;

use tk_convert::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

/// The proven fold emits a vocabulary entry without merging, so it is only valid if the merge
/// loop would have produced that same entry. gpt2 does not declare `ignore_merges`, so here
/// the fold is on purely because the proof enabled it -- which makes it the config where a
/// wrong proof would show up.
///
/// These strings mix words that are a single vocabulary entry (folded) with words that are
/// not (merged), and include the special token whose entry does NOT fold: `<|endoftext|>`
/// decomposes to seven tokens, and folding it would emit one.
#[test]
fn the_proven_fold_never_changes_the_ids() {
    let reference = Tokenizer::from_file("../data/gpt2.json").unwrap();
    let pipe = PipelineTokenizer::try_from(&reference).unwrap();

    for text in [
        " the quick brown fox jumps over the lazy dog",
        "unprefixed words and internationalisation",
        "def foo(bar):\n    return bar + 1\n",
        "<|endoftext|> literal in the middle <|endoftext|>",
        " 语言模型 mixed with ASCII and ελληνικά",
        "aaaaaaaaaaaaaaaaaaaaaaaa",
        "   ",
        "",
    ] {
        let want: Vec<u32> = reference
            .encode_fast(text, false)
            .unwrap()
            .get_ids()
            .to_vec();
        let got: Vec<u32> = pipe
            .encode(text, false)
            .wait()
            .unwrap()
            .remove(0)
            .ids()
            .iter()
            .map(|t| t.id())
            .collect();
        assert_eq!(want, got, "the fold changed the ids for {text:?}");
    }
}

/// `PipelineBPE::tokenize_spans` is an override of a trait method whose default is the
/// `tokenize_pipeline` loop, so the two can drift apart without anything failing to build --
/// which is how it came to destructure a `BpeScratch` that no longer had those fields.
///
/// The short strings above pass through the batch loop a handful of spans at a time. This one
/// gives it thousands in a single chunk, with the traffic that separates the two paths:
/// repeats (so the word cache both fills and hits), words the fold serves, words that must
/// merge, punctuation runs, multi-byte scripts, and a long unbroken run.
#[test]
fn the_batched_path_matches_the_reference() {
    let reference = Tokenizer::from_file("../data/gpt2.json").unwrap();
    let pipe = PipelineTokenizer::try_from(&reference).unwrap();

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

    let want: Vec<u32> = reference
        .encode_fast(text.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let got: Vec<u32> = pipe
        .encode(text.as_str(), false)
        .wait()
        .unwrap()
        .remove(0)
        .ids()
        .iter()
        .map(|t| t.id())
        .collect();
    assert_eq!(want.len(), got.len(), "token count differs");
    assert_eq!(want, got, "the batched path changed the ids");
}

/// The gate that matters for decode: encode is compared by `json_oracle`, but nothing there
/// exercises the decoder, so compare both paths' `decode` on the real files.
///
/// Only the non-byte-level models, because [`PipelineTokenizer::decode`] short-circuits a
/// byte-level BPE and never consults the decoder at all — for those the wiring is untestable
/// from here, and identical by construction.
#[test]
fn decode_matches_the_config_path_on_the_real_configs() {
    let ids: Vec<u32> = vec![10, 200, 3000, 7, 42, 1, 0, 999];
    for file in ["llama-2.json", "t5-base.json", "bert-base-uncased.json"] {
        let path = format!("../data/{file}");
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let declared = Tokenizer::from_file(&path).unwrap();
        let config_path = PipelineTokenizer::try_from(&declared).unwrap();
        let slim = tk_serialize::from_json_file(&path).unwrap();
        for skip_special in [false, true] {
            assert_eq!(
                slim.decode(&ids, skip_special).unwrap(),
                config_path.decode(&ids, skip_special).unwrap(),
                "{file} skip_special={skip_special}"
            );
        }
    }
}
