// Byte-level bypass: pipeline-level eligibility + fast-path/slow-path equivalence.
// Excluded on windows, which does not download the *-slim.json test fixtures.
#![cfg(not(target_os = "windows"))]

use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{AddedToken, Tokenizer, TokenizerTrainExt, Trainable};

fn load(f: &str) -> Tokenizer {
    Tokenizer::from_file(format!("data/{f}")).unwrap()
}

// --- Eligibility: which loaded pipelines activate the bypass on deserialize ---

#[test]
fn enabled_for_byte_level_models() {
    // top-level ByteLevel: gpt2, roberta | nested in a Sequence: deepseek, llama3, glm, gpt-oss
    for config_file in [
        "gpt2-slim.json",
        "roberta-slim.json",
        "deepseek-v4-slim.json",
        "llama-3-slim.json",
        "glm-5.2-slim.json",
        "gpt-oss-slim.json",
    ] {
        assert!(
            load(config_file).byte_level_bypass_enabled(),
            "{} must enable the bypass path",
            config_file
        );
    }
}

#[test]
fn disabled_for_non_byte_level_models() {
    for config_file in ["gemma-4-slim.json", "bert-wiki-slim.json"] {
        assert!(
            !load(config_file).byte_level_bypass_enabled(),
            "{} must NOT enable the bypass path",
            config_file
        );
    }
}

#[test]
fn empty_sequence_normalizer_counts_as_noop() {
    // deepseek's normalizer is Sequence[] — must not disqualify
    assert!(load("deepseek-v4-slim.json").byte_level_bypass_enabled());
}

#[test]
fn disabled_when_pretokenizer_swapped_out() {
    let mut tok = load("gpt2-slim.json");
    assert!(tok.byte_level_bypass_enabled());
    tok.with_pre_tokenizer(Some(Whitespace)); // auto-refresh
    assert!(!tok.byte_level_bypass_enabled());
}

#[test]
fn disabled_when_real_normalizer_added() {
    let mut tok = load("deepseek-v4-slim.json");
    assert!(tok.byte_level_bypass_enabled());
    tok.with_normalizer(Some(NFC)).unwrap(); // auto-refresh
    assert!(!tok.byte_level_bypass_enabled());
}

// --- Equivalence: fast path (forced on) must equal slow path (forced off) ---

const CORPUS: &[&str] = &[
    "",
    " ",
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "   leading and trailing   ",
    "café naïve Ωμέγα",
    "日本語のテキストをトークン化する",
    "emoji 👍🇫🇷👨\u{200d}👩\u{200d}👧 mixed",
    "tabs\tand\nnewlines\n\n",
    "numbers 1234567890 and symbols !@#$%^&*()",
    "internationalization tokenization preprocessing",
    "Mr. O'Brien's well-thought-out, multi-part résumé.",
    // Single vocab entries in the ignore_merges fixtures whose greedy-merge
    // segmentation differs from the whole token:
    // "ato" (llama-3), "conf" (glm-5.2), "rg" (gpt-oss).
    "ato",
    "conf",
    "rg",
];

/// Compare the fast path (forced on) against the slow path (forced off) on the same
/// tokenizer: `encode` ids+offsets+tokens and `encode_fast` ids.
fn assert_fast_matches_slow(config_file: &str) {
    let mut tok = load(config_file);
    assert!(
        tok.byte_level_bypass_enabled(),
        "{} must be eligible for the bypass path",
        config_file
    );
    for &text in CORPUS {
        tok.set_byte_level_bypass(true);
        let fast = tok.encode(text, false).unwrap();
        let fast_ids = fast.get_ids().to_vec();
        let fast_offsets = fast.get_offsets().to_vec();
        let fast_tokens = fast.get_tokens().to_vec();
        let fast_no_offsets = tok.encode_fast(text, false).unwrap().get_ids().to_vec();

        tok.set_byte_level_bypass(false);
        let slow = tok.encode(text, false).unwrap();
        let slow_ids = slow.get_ids().to_vec();
        let slow_offsets = slow.get_offsets().to_vec();
        let slow_tokens = slow.get_tokens().to_vec();
        let slow_no_offsets = tok.encode_fast(text, false).unwrap().get_ids().to_vec();

        assert_eq!(
            fast_ids, slow_ids,
            "encode ids differ — {config_file} on {text:?}"
        );
        assert_eq!(
            fast_offsets, slow_offsets,
            "encode offsets differ — {config_file} on {text:?}"
        );
        assert_eq!(
            fast_tokens, slow_tokens,
            "encode tokens differ — {config_file} on {text:?}"
        );
        assert_eq!(
            fast_no_offsets, slow_no_offsets,
            "encode_fast ids differ — {config_file} on {text:?}"
        );
    }
}

#[test]
fn encode_matches_slow_path() {
    for config_file in [
        "gpt2-slim.json",
        "roberta-slim.json",
        "deepseek-v4-slim.json",
    ] {
        assert_fast_matches_slow(config_file);
    }
}

#[test]
fn encode_matches_slow_path_with_ignore_merges() {
    for config_file in [
        "llama-3-slim.json",
        "glm-5.2-slim.json",
        "gpt-oss-slim.json",
    ] {
        assert_fast_matches_slow(config_file);
    }
}

#[test]
fn encode_batch_matches_slow_path() {
    // A large batch drives the rayon path, exercising the per-thread byte cache.
    let mut tok = load("gpt2-slim.json");
    let batch: Vec<&str> = CORPUS.iter().cloned().cycle().take(512).collect();

    tok.set_byte_level_bypass(true);
    let fast = tok.encode_batch(batch.clone(), false).unwrap();
    tok.set_byte_level_bypass(false);
    let slow = tok.encode_batch(batch, false).unwrap();

    for (i, (f, s)) in fast.iter().zip(&slow).enumerate() {
        assert_eq!(f.get_ids(), s.get_ids(), "batch ids differ at {i}");
        assert_eq!(
            f.get_offsets(),
            s.get_offsets(),
            "batch offsets differ at {i}"
        );
    }
}

#[test]
fn encode_with_added_tokens_matches_slow_path() {
    let mut tok = load("gpt2-slim.json");
    tok.add_special_tokens([AddedToken::from("<|endoftext|>", true)])
        .unwrap();
    tok.add_tokens([AddedToken::from("[CUSTOM]", false)])
        .unwrap();

    let corpus = [
        "<|endoftext|>hello world<|endoftext|>",
        "before [CUSTOM] after",
        "[CUSTOM]<|endoftext|> 日本 👍 [CUSTOM]",
        "no special tokens here",
    ];
    for text in corpus {
        tok.set_byte_level_bypass(true);
        let fast = tok.encode(text, true).unwrap();
        tok.set_byte_level_bypass(false);
        let slow = tok.encode(text, true).unwrap();

        assert_eq!(fast.get_ids(), slow.get_ids(), "ids differ on {text:?}");
        assert_eq!(
            fast.get_offsets(),
            slow.get_offsets(),
            "offsets differ on {text:?}"
        );
        assert_eq!(
            fast.get_tokens(),
            slow.get_tokens(),
            "tokens differ on {text:?}"
        );
    }
}

#[test]
fn encode_matches_slow_path_on_big_corpus() {
    let mut tok = load("gpt2-slim.json");
    assert!(tok.byte_level_bypass_enabled());
    let text = std::fs::read_to_string("data/big.txt").unwrap();
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();

    tok.set_byte_level_bypass(true);
    let fast = tok.encode_batch(lines.clone(), false).unwrap();
    tok.set_byte_level_bypass(false);
    let slow = tok.encode_batch(lines.clone(), false).unwrap();

    for (line, (f, s)) in lines.iter().zip(fast.iter().zip(&slow)) {
        assert_eq!(f.get_ids(), s.get_ids(), "ids differ on {line:?}");
        assert_eq!(
            f.get_offsets(),
            s.get_offsets(),
            "offsets differ on {line:?}"
        );
    }
}

#[test]
fn encode_after_training_matches_slow_path() {
    let mut tok = load("gpt2-slim.json");
    assert!(tok.byte_level_bypass_enabled());

    let mut trainer = tok.get_model().get_trainer();
    tok.train_from_files(&mut trainer, vec!["data/small.txt".to_string()])
        .unwrap();

    for text in [
        "hello world",
        "The quick brown fox",
        "café 日本 👍",
        " a b c",
    ] {
        let fast = tok.encode(text, false).unwrap().get_ids().to_vec();
        tok.set_byte_level_bypass(false);
        let slow = tok.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(fast, slow, "post-train encode diverges on {text:?}");
    }
}
