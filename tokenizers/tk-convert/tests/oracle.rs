//! The pipeline must encode and decode exactly like the latest *released* `tokenizers`.
//!
//! Each model's `tokenizer.json` is a fixture pinned in `hf-internal-testing/tokenizers-test-data`
//! (`make oracle` fetches them), converted by this crate, and read back by the canonical reader --
//! the pairing tk-convert exists to make work. The release is the oracle, so nothing in this tree
//! grades its own homework.
//!
//! One test per model, covering the three shapes the conversion has to handle: byte-level BPE
//! (gpt2, llama-3.2-1b), WordPiece (bert-base-uncased), and SentencePiece Unigram (t5-base,
//! albert-base-v1). `meta-llama/Llama-3.2-1B` is gated, so its fixture is a mirrored copy
//! (hf-internal-testing/tokenizers-test-data#10) rather than the Hub repo itself.
//!
//! all-MiniLM-L6-v2 and all-mpnet-base-v2 are WordPiece again, but their configs ship `truncation`
//! and `padding` settings, so they are where the padded attention mask and type ids are compared.
//!
//! siglip-base-patch16-224 is a lone `Metaspace`, where t5-base and albert-base-v1 pair it with a
//! `WhitespaceSplit`, and the two shapes convert differently. mistral-7b-v0.1 sets
//! `prepend_scheme: first` with `split: false`.
//!
//! Decode is fed the release's *own* ids, so it is judged on decode alone even where encode
//! legitimately diverges.
//!
//!   make oracle

#![cfg(feature = "bench-baseline")]

use tk_convert::ConvertError;
use tk_encode::pipeline::EncodeOptions;
use tokenizers_release::Tokenizer as Released;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

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

fn assert_matches_released(repo: &str, file: &str) {
    let path = format!("{DATA}/{file}");
    let canonical = match tk_convert::canonicalize_file(&path) {
        Ok(c) => c,
        Err(ConvertError::Io { .. }) => panic!(
            "{repo}: no fixture at {path}. Run `make oracle` to fetch it, or add it to \
             hf-internal-testing/tokenizers-test-data if it isn't there yet."
        ),
        // A refusal here is the regression this oracle exists to catch, so it fails, not skips.
        Err(e) => panic!("{repo}: this pass refuses it: {e}"),
    };
    let pipeline = tk_serialize::from_json(&canonical)
        .unwrap_or_else(|e| panic!("{repo}: the canonical reader refuses the conversion: {e}"));
    let released = Released::from_file(&path).expect("the released crate reads it");

    let mut diverged = Vec::new();
    for text in &cases() {
        let text = text.as_str();
        for options in [EncodeOptions::no_specials(), EncodeOptions::default()] {
            let want = released.encode_fast(text, options.add_special_tokens).unwrap();
            let ids = want.get_ids().to_vec();
            let encodings = pipeline.encode(text, &options).wait().unwrap();
            let encoding = &encodings[0];
            let got: Vec<u32> = encoding.ids().iter().map(|t| t.id()).collect();
            if ids != got {
                diverged.push(format!("encode options={options:?} {text:?}"));
                continue; // decoding ids we already disagree about says nothing
            }
            let mask: Vec<u32> = match encoding.attention_mask() {
                Some(mask) => mask.iter().copied().map(u32::from).collect(),
                None => vec![1; got.len()],
            };
            if mask != want.get_attention_mask() {
                diverged.push(format!("attention_mask options={options:?} {text:?}"));
            }
            let type_ids: Vec<u32> = match encoding.type_ids() {
                Some(type_ids) => type_ids.iter().copied().map(u32::from).collect(),
                None => vec![0; got.len()],
            };
            if type_ids != want.get_type_ids() {
                diverged.push(format!("type_ids options={options:?} {text:?}"));
            }
            for skip in [false, true] {
                let decoded = released.decode(&ids, skip).unwrap();
                if pipeline.decode(&ids, skip).unwrap_or_default() != decoded {
                    diverged.push(format!("decode skip={skip} {text:?}"));
                }
            }
        }
    }
    assert!(
        diverged.is_empty(),
        "{repo} diverges from the released crate:\n{}",
        diverged.join("\n")
    );
}

#[test]
fn gpt2() {
    assert_matches_released("gpt2", "gpt2.json");
}

#[test]
fn bert_base_uncased() {
    assert_matches_released("bert-base-uncased", "bert-base-uncased.json");
}

#[test]
fn t5_base() {
    assert_matches_released("t5-base", "t5-base.json");
}

#[test]
fn albert_base_v1() {
    assert_matches_released("albert-base-v1", "albert-base-v1-tokenizer.json");
}

#[test]
fn llama_3_2_1b() {
    assert_matches_released(
        "meta-llama/Llama-3.2-1B",
        "fixtures/models/llama-3.2-1b.json",
    );
}

#[test]
#[ignore] // TODO: un-skip this once we ship truncation
fn all_minilm_l6_v2() {
    assert_matches_released(
        "sentence-transformers/all-MiniLM-L6-v2",
        "fixtures/models/all-minilm-l6-v2.json",
    );
}

#[test]
#[ignore] // TODO: un-skip this once we ship truncation
fn all_mpnet_base_v2() {
    assert_matches_released(
        "sentence-transformers/all-mpnet-base-v2",
        "fixtures/models/all-mpnet-base-v2.json",
    );
}

#[test]
fn siglip_base_patch16_224() {
    assert_matches_released(
        "google/siglip-base-patch16-224",
        "fixtures/models/siglip-base-patch16-224.json",
    );
}

#[test]
fn mistral_7b_v0_1() {
    assert_matches_released(
        "mistralai/Mistral-7B-v0.1",
        "fixtures/models/mistral-7b-v0.1.json",
    );
}
