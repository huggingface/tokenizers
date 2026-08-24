//! The pipeline must encode and decode exactly like the latest *released* `tokenizers`.
//!
//! Each model is fetched from the Hub, converted by this crate, and read back by the canonical
//! reader -- the pairing tk-convert exists to make work. The release is the oracle, so nothing in
//! this tree grades its own homework. `hf-hub` caches, so only the first run is online.
//!
//! Decode is fed the release's *own* ids, so it is judged on decode alone even where encode
//! legitimately diverges.
//!
//!   cargo test -p tk-convert --features bench-baseline --test oracle

#![cfg(feature = "bench-baseline")]

use tokenizers_release::Tokenizer as Released;

/// Byte-level BPE, WordPiece, and SentencePiece Unigram -- the three shapes the conversion has to
/// handle.
const MODELS: &[&str] = &[
    "gpt2",
    "bert-base-uncased",
    "t5-base",
    "albert-base-v1",
    "meta-llama/Llama-3.2-1B",
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
    for &repo in MODELS {
        let path = match hf_hub::api::sync::Api::new()
            .and_then(|api| api.model(repo.to_string()).get("tokenizer.json"))
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skip {repo}: cannot fetch tokenizer.json ({e})");
                continue;
            }
        };
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
