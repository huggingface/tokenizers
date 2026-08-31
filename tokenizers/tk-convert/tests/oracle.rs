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
    // A lone `Metaspace`. t5 and albert are the `Sequence[WhitespaceSplit, Metaspace]` pair, and
    // the two shapes convert differently.
    "google/siglip-base-patch16-224",
    // `prepend_scheme: first` with `split: false`.
    "mistralai/Mistral-7B-v0.1",
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
    // Only U+0020 becomes a delimiter. Tabs, newlines and the other space characters survive.
    "",
    " ",
    "   ",
    "\t",
    "\n",
    " \t\n ",
    "a  b   c",
    "\tleading tab",
    "trailing space ",
    "a\u{a0}b",
    "a\u{3000}b",
    "a\nb\tc\r\nd",
    // A delimiter already in the input. `prepend` must not add a second one.
    "▁",
    "▁▁",
    "▁leading",
    "a▁b",
    "a▁ b▁c",
    // Added tokens cut the sequence into segments. `first` writes the delimiter only on the
    // segment at offset zero, so the token's position decides the ids.
    "hello</s>world",
    "</s>tail",
    "head</s>",
    "</s>",
    "</s></s>",
    "</s></s>x",
    "a </s> b",
    "a</s> b",
    "a </s>b",
    "<unk>x",
    "</s> ▁mixed </s>",
    // Codepoints that normalization moves.
    "café",
    "cafe\u{301}",
    "👨\u{200d}👩\u{200d}👧\u{200d}👦 café",
    "\u{feff}bom",
    "\u{301}",
    "i\u{307}\u{323}",
];

/// [`TEXTS`] plus one input long enough for the parallel encoder to take over.
///
/// Both halves clear `PARALLEL_MIN_BYTES`, so the planner splits the sequence at the added token
/// and encodes each chunk on its own.
fn cases() -> Vec<String> {
    let mut out: Vec<String> = TEXTS.iter().map(|s| (*s).to_string()).collect();
    let side = |word: &str| {
        let repeats = 2 * tk_encode::pipeline::PARALLEL_MIN_BYTES / (word.len() + 1);
        word.to_string() + &format!(" {word}").repeat(repeats)
    };
    out.push(format!("{}</s>{}", side("alpha"), side("beta")));
    out
}

fn fetch(repo: &str) -> Option<std::path::PathBuf> {
    match hf_hub::api::sync::Api::new()
        .and_then(|api| api.model(repo.to_string()).get("tokenizer.json"))
    {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("skip {repo}: cannot fetch tokenizer.json ({e})");
            None
        }
    }
}

#[test]
fn matches_the_released_crate() {
    let mut diverged = Vec::new();
    for &repo in MODELS {
        let Some(path) = fetch(repo) else { continue };
        let (repo, path) = (&repo, &path);
        let canonical = tk_convert::canonicalize_file(path)
            .unwrap_or_else(|e| panic!("{repo}: this pass refuses it: {e}"));
        // A refusal here is the regression this oracle exists to catch, so it fails, not skips.
        let pipeline = tk_serialize::from_json(&canonical)
            .unwrap_or_else(|e| panic!("{repo}: the canonical reader refuses the conversion: {e}"));
        let released = Released::from_file(path).expect("the released crate reads it");

        for text in &cases() {
            let text = text.as_str();
            for special in [false, true] {
                let ids = released
                    .encode_fast(text, special)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                let got: Vec<u32> = pipeline.encode(text, special).wait().unwrap()[0]
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
