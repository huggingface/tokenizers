//! Does canonicalizing a config change what it encodes to?
//!
//! One gate, over every real config in `data/`: read the file with the slim reader directly, read it
//! again through `tk_convert::canonicalize_str`, and demand identical ids from both. The converter
//! is a JSON→JSON rewrite, so "it rewrote something" is trivially true and proves nothing; the only
//! values it *perturbs* are Unigram scores, which is why the test insists a Unigram config was among
//! those compared.
//!
//! The file used to hold four more tests, all of which used `tk_convert::Tokenizer` -- the serde
//! config reader -- as the oracle the slim reader was checked against: a byte-level fold check, a
//! batched-path check, and a decode and an encode comparison over every real config (266
//! comparisons across 19 configs). That reader no longer exists, and neither do they. What is left
//! proves the *converter* against the reader, not the reader against a second implementation; the
//! independent oracle is `tests/pipeline_oracle.rs`, which compares against the released crate.
//! `REQUIRED_FOR_V1.md` §7 has the honest accounting.

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
        // `data/tokenizer.json` is not a fixture: `tests/documentation.rs` *writes* it with
        // `.save("data/tokenizer.json", ..)`. It is a 100-token, 20-merge toy missing byte atoms, so
        // the pipeline rightly refuses it -- but cargo runs test binaries in parallel, so reading it
        // here can race that write. Excluded by name rather than left to fail as a confusing skip.
        .filter(|p| p.file_name().is_some_and(|n| n != "tokenizer.json"))
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
        // Read the tag off the *canonical* form, not the raw file. `albert-base-v1-tokenizer.json`
        // is a Unigram model whose `model` object has no `"type"` at all -- inferring it is one of
        // the things canonicalize does -- so looking in the raw text misses exactly the legacy
        // files this test most wants to cover.
        if canonical.contains("\"Unigram\"") {
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
