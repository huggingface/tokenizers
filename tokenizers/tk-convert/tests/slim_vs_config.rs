//! Comparisons between the slim JSON reader and this crate's serde reader.
//!
//! Every one builds the pipeline both ways from the *same* file and demands identical output: the
//! byte-level merge fold's ids against the reference `Tokenizer`, then `decode`, then `encode` over
//! every real config in `data/`.
//!
//! The encode comparison deliberately holds no recorded baseline. An earlier version of this gate
//! was an example carrying a digest file, and the digests went stale against a newer fixture --
//! every run "passed" while comparing against ids no longer produced. Comparing the two code paths
//! against each other in-process has nothing to go stale.

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

/// The gate that matters for decode: `encode_matches_the_config_path_on_every_real_config` covers
/// encode, but nothing there exercises the decoder, so compare both paths' `decode` too.
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

/// Encode, both paths, every real config in `data/`.
///
/// This is the comparison that pins ids: the slim reader constructs `PipelineTokenizer` directly
/// while the config path lowers a deserialized `Tokenizer`, and the two constructions have to agree
/// token for token. Replaying added tokens in the wrong order against the wrong model is the one
/// mistake that moves ids silently, and it is only visible here.
///
/// The fixture list is read from the directory rather than hard-coded so a new `make models` file is
/// covered the day it lands. A config either path refuses is skipped and named on stderr -- the
/// refusals are real (`gpt2-vocab.json` is a bare vocab, not a tokenizer; some configs need a regex
/// engine this crate's dev-dependencies do not turn on) -- and the `>= 1` guard is only there to
/// stop the whole thing passing vacuously when `data/` is empty, as it is on the Windows CI leg.
#[test]
fn encode_matches_the_config_path_on_every_real_config() {
    let texts = [
        " the quick brown fox jumps over the lazy dog",
        "def foo(bar):\n    return bar + 1\n",
        " 语言模型 mixed with ASCII and ελληνικά",
        "الْعَرَبِيَّة и русский текст",
        "unprefixed internationalisation",
        "   ",
        "",
    ];

    let dir = std::path::Path::new("../data");
    if !dir.exists() {
        return;
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read data/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();

    let mut compared = 0usize;
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // The reason is printed, not swallowed: a skip is a comparison not made, and "refused it"
        // without a cause is how a config quietly stops being covered.
        let declared = match Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skip {name}: the serde reader refused it: {e}");
                continue;
            }
        };
        let config_path = match PipelineTokenizer::try_from(&declared) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skip {name}: it does not lower to a pipeline: {e}");
                continue;
            }
        };
        let slim = match tk_serialize::from_json_file(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skip {name}: the slim reader refused it: {e}");
                continue;
            }
        };

        for text in texts {
            for add_special in [false, true] {
                let ids = |t: &PipelineTokenizer| -> Vec<u32> {
                    t.encode(text, add_special)
                        .wait()
                        .unwrap()
                        .remove(0)
                        .ids()
                        .iter()
                        .map(|t| t.id())
                        .collect()
                };
                assert_eq!(
                    ids(&config_path),
                    ids(&slim),
                    "{name} add_special={add_special} text={text:?}"
                );
                compared += 1;
            }
        }
    }

    assert!(
        compared >= 1,
        "../data exists but nothing was comparable, so this asserted nothing"
    );
    eprintln!("compared {compared} encode(s) across both readers");
}

/// Canonicalizing first must not move ids -- the path with a known float perturbation.
///
/// `canonicalize` parses with `serde_json`, whose default float path is 1 ULP wrong on some
/// literals, then writes the shortest form of the double it got. So a round-trip through it shifts
/// about a quarter of a Unigram model's scores by exactly one ULP:
///
/// ```text
/// -3.8403830528259277  ->  -3.840383052825928     c00eb91ac0000000 -> c00eb91ac0000001
/// -9.008880615234375   ->  -9.008880615234377     c022048c00000000 -> c022048c00000001
/// ```
///
/// The second is a dyadic rational, exact in binary, and it still moves -- so this is a real change
/// of value, not a reformat. It is `serde_json`'s documented behaviour rather than a defect here, and
/// `float_roundtrip` is not the fix: it was measured to change shipped ids.
///
/// A one-ULP change to a log-probability almost never flips a Viterbi path, which is exactly why this
/// needs a gate rather than an argument -- "almost never" is not "never", and t5 and albert are the
/// two configs where it could bite.
#[test]
fn canonicalizing_first_does_not_move_ids() {
    let texts = [
        " the quick brown fox jumps over the lazy dog",
        "unprefixed internationalisation and tokenisation",
        " 语言模型 mixed with ASCII",
        "The Sphinx of black quartz, judge my vow.",
    ];

    let dir = std::path::Path::new("../data");
    if !dir.exists() {
        return;
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read data/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();

    let mut compared = 0usize;
    // Whether a Unigram config was among those compared. Byte-inequality would be useless here --
    // canonicalize reformats every file it accepts, so "it rewrote something" is trivially true. The
    // scores are the only values it perturbs, so a Unigram model is what makes this test mean
    // anything; without the guard it would keep passing if `unigram` ever left the dev-dependencies,
    // silently retiring the case it exists for.
    let mut unigram_seen = 0usize;
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Only the files both paths accept; the refusals are covered by the encode gate above.
        let Ok(direct) = tk_serialize::from_json(&text) else {
            continue;
        };
        let canonical = match tk_convert::canonicalize_str(&text) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  skip {name}: canonicalize refused it: {e}");
                continue;
            }
        };
        let via_canonical = match tk_serialize::from_json(&canonical) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skip {name}: the reader refused the canonical form: {e}");
                continue;
            }
        };
        if text.contains("\"Unigram\"") {
            unigram_seen += 1;
        }

        for text in texts {
            for add_special in [false, true] {
                let ids = |t: &tk_encode::pipeline::PipelineTokenizer| -> Vec<u32> {
                    t.encode(text, add_special)
                        .wait()
                        .unwrap()
                        .remove(0)
                        .ids()
                        .iter()
                        .map(|t| t.id())
                        .collect()
                };
                assert_eq!(
                    ids(&direct),
                    ids(&via_canonical),
                    "{name} moved ids through canonicalize: add_special={add_special} text={text:?}"
                );
                compared += 1;
            }
        }
    }

    assert!(
        compared >= 1,
        "../data exists but nothing was comparable, so this asserted nothing"
    );
    assert!(
        unigram_seen >= 1,
        "no Unigram config was compared, so the score round-trip -- the only thing canonicalize \
         perturbs -- was never exercised"
    );
    eprintln!(
        "compared {compared} encode(s) raw vs canonicalized ({unigram_seen} Unigram config(s))"
    );
}
