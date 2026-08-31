//! The pipeline must encode and decode exactly like the latest *released* `tokenizers`.
//!
//! Each model's `tokenizer.json` is a fixture pinned in `hf-internal-testing/tokenizers-test-data`
//! (`make oracle` fetches them), converted by this crate, and read back by the canonical reader --
//! the pairing tk-convert exists to make work. The release is the oracle, so nothing in this tree
//! grades its own homework.
//!
//! Decode is fed the release's *own* ids, so it is judged on decode alone even where encode
//! legitimately diverges.
//!
//!   make oracle

#![cfg(feature = "bench-baseline")]

use tokenizers_release::Tokenizer as Released;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

/// (Hub repo, its `tokenizer.json` fixture) pairs, covering the three shapes the conversion has
/// to handle: byte-level BPE, WordPiece, and SentencePiece Unigram.
const MODELS: &[(&str, &str)] = &[
    ("gpt2", "gpt2.json"),
    ("bert-base-uncased", "bert-base-uncased.json"),
    ("t5-base", "t5-base.json"),
    ("albert-base-v1", "albert-base-v1-tokenizer.json"),
];

/// One line per script the old fixture corpora covered, plus the two modalities that stress the
/// byte and delimiter paths hardest. Short on purpose -- these exercise the encoders' branches, not
/// their throughput.
const TEXTS: &[&str] = &[
    "The quick brown fox jumps 123.",
    " héllo wörld ",                     // accents, leading and trailing space
    "你好世界",                          // Han
    "こんにちは世界",                    // Japanese, mixed scripts
    "Привет мир",                        // Cyrillic
    "مرحبا بالعالم",                     // Arabic, RTL
    "नमस्ते दुनिया",                        // Devanagari, combining marks
    "வணக்கம் உலகம்",                        // Tamil
    "สวัสดีชาวโลก",                        // Thai, no word spaces
    "ሰላም ዓለም",                           // Ethiopic
    "fn main() { let x = vec![1, 2]; }", // code
    r"\frac{1}{2} \sum_{i=0}^{n}",       // math
];

#[test]
fn matches_the_released_crate() {
    let mut diverged = Vec::new();
    for &(repo, file) in MODELS {
        let path = format!("{DATA}/{file}");
        let canonical = tk_convert::canonicalize_file(&path)
            .unwrap_or_else(|e| panic!("{repo}: this pass refuses it: {e}"));
        // A refusal here is the regression this oracle exists to catch, so it fails, not skips.
        let pipeline = tk_serialize::from_json(&canonical)
            .unwrap_or_else(|e| panic!("{repo}: the canonical reader refuses the conversion: {e}"));
        let released = Released::from_file(&path).expect("the released crate reads it");

        for text in TEXTS {
            for special in [false, true] {
                let ids = released
                    .encode_fast(*text, special)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                let got: Vec<u32> = pipeline.encode(*text, special).wait().unwrap()[0]
                    .ids()
                    .iter()
                    .map(|t| t.id())
                    .collect();
                if ids != got {
                    diverged.push(format!("{repo} encode special={special} {text:?}"));
                    continue; // decoding ids we already disagree about says nothing
                }
                for skip in [false, true] {
                    let want = released.decode(&ids, skip).unwrap();
                    if pipeline.decode(&ids, skip).unwrap_or_default() != want {
                        diverged.push(format!("{repo} decode skip={skip} {text:?}"));
                    }
                }
            }
        }
    }
    assert!(
        diverged.is_empty(),
        "diverges from the released crate:\n{}",
        diverged.join("\n")
    );
}
