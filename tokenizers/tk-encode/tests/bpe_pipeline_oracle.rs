//! `PipelineTokenizer` must produce the same ids as the legacy `Tokenizer` for every BPE model we
//! ship a `tokenizer.json` for. The legacy engine is the oracle: it is the code main runs.
//!
//! This covers ground the bert-wiki oracle cannot. In particular llama-2 is the only model here that
//! is not byte-level -- it takes the `Atoms::Chars` path with `byte_fallback`, `fuse_unk` and a
//! space-rewriting normalizer -- and llama-2 and llama-3 are the only ones with merges that are
//! unsafe to batch, which is what the `SAFE` flag in the pair table exists for: ~22% of their merges
//! have a product that can reach a cheaper merge, so a multipass sweep that merged every occurrence
//! of the min pair at once would diverge from BPE order. gpt2 and deepseek have none.
use std::convert::TryFrom;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const MODELS: &[(&str, &str)] = &[
    ("gpt2", "../data/gpt2.json"),
    ("llama-3", "../data/llama-3-tokenizer.json"),
    ("deepseek", "../data/deepseek-v4.json"),
    ("llama-2", "../data/llama-2.json"),
];

const CORPORA: &[(&str, &str)] = &[
    ("english", "../data/big.txt"),
    ("japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("code", "../data/corpora/code.txt"),
    ("greek", "../data/corpora/greek.txt"),
    ("russian", "../data/corpora/russian.txt"),
    ("korean", "../data/corpora/korean.txt"),
    ("arabic", "../data/corpora/arabic.txt"),
    ("hindi", "../data/corpora/hindi.txt"),
    ("thai", "../data/corpora/thai.txt"),
    ("chinese", "../data/corpora/chinese.txt"),
];

/// Enough to exercise long pre-tokens on both sides of the gate without making the suite slow.
const PER_CORPUS_BYTES: usize = 400_000;
const CHUNK_BYTES: usize = 4096;

fn check_model(name: &str, path: &str) {
    let Ok(oracle) = Tokenizer::from_file(path) else {
        eprintln!("bpe oracle: skip {name} -- {path} not found");
        return;
    };
    let pipeline = PipelineTokenizer::try_from(&oracle)
        .unwrap_or_else(|e| panic!("{name}: pipeline construction failed: {e}"));

    let mut checked = 0usize;
    for (corpus, corpus_path) in CORPORA {
        let Ok(text) = std::fs::read_to_string(corpus_path) else {
            continue;
        };
        let mut end = PER_CORPUS_BYTES.min(text.len());
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        let mut chunk = String::new();
        for line in text[..end].lines().filter(|l| !l.trim().is_empty()) {
            chunk.push('\n');
            chunk.push_str(line);
            if chunk.len() < CHUNK_BYTES {
                continue;
            }
            let expected = oracle.encode(chunk.as_str(), false).unwrap();
            let got: Vec<u32> = pipeline
                .encode(&chunk, false)
                .wait()
                .unwrap()
                .first()
                .unwrap()
                .ids()
                .iter()
                .map(|t| t.id())
                .collect();
            assert_eq!(
                expected.get_ids(),
                got.as_slice(),
                "{name} / {corpus}: id mismatch on {:?}",
                chunk.chars().take(80).collect::<String>()
            );
            checked += chunk.len();
            chunk.clear();
        }
    }
    assert!(checked > 100_000, "{name}: only {checked} bytes checked");
    println!("{name}: {checked} bytes byte-exact vs legacy");
}

#[test]
fn gpt2_matches_legacy() {
    check_model("gpt2", MODELS[0].1);
}

#[test]
fn llama_3_matches_legacy() {
    check_model("llama-3", MODELS[1].1);
}

#[test]
fn deepseek_matches_legacy() {
    check_model("deepseek", MODELS[2].1);
}

#[test]
fn llama_2_matches_legacy() {
    check_model("llama-2", MODELS[3].1);
}

fn check_model_parallel(name: &str, path: &str) {
    const CHUNK_SIZES: &[usize] = &[12 * 1024, 32 * 1024, 96 * 1024];

    let Ok(oracle) = Tokenizer::from_file(path) else {
        eprintln!("bpe parallel oracle: skip {name} -- {path} not found");
        return;
    };
    let pipeline = PipelineTokenizer::try_from(&oracle)
        .unwrap_or_else(|e| panic!("{name}: pipeline construction failed: {e}"));

    let mut checked = 0usize;
    let mut size_idx = 0usize;
    for (corpus, corpus_path) in CORPORA {
        let Ok(text) = std::fs::read_to_string(corpus_path) else {
            continue;
        };
        let mut end = PER_CORPUS_BYTES.min(text.len());
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        let mut chunk = String::new();
        let mut target = CHUNK_SIZES[size_idx % CHUNK_SIZES.len()];
        for line in text[..end].lines().filter(|l| !l.trim().is_empty()) {
            chunk.push('\n');
            chunk.push_str(line);
            if chunk.len() < target {
                continue;
            }
            let expected = oracle.encode(chunk.as_str(), false).unwrap();
            let got: Vec<u32> = pipeline
                .encode(chunk.as_str(), false)
                .wait()
                .unwrap()
                .first()
                .unwrap()
                .ids()
                .iter()
                .map(|t| t.id())
                .collect();
            assert_eq!(
                expected.get_ids(),
                got.as_slice(),
                "{name} / {corpus} (parallel, {} bytes): id mismatch on {:?}",
                chunk.len(),
                chunk.chars().take(80).collect::<String>()
            );
            checked += chunk.len();
            chunk.clear();
            size_idx += 1;
            target = CHUNK_SIZES[size_idx % CHUNK_SIZES.len()];
        }
    }
    assert!(
        checked > 100_000,
        "{name}: only {checked} bytes checked on the parallel path"
    );
    println!("{name}: {checked} bytes byte-exact vs legacy (parallel path)");
}

#[test]
fn gpt2_parallel_matches_legacy() {
    check_model_parallel("gpt2", MODELS[0].1);
}

/// Lightweight smoke: pairs still encode correctly in a BPE context (serial path — the parallel
/// pair path is covered by the unit `parallel_matches_serial_pairs_*` tests). Just confirms the
/// pipeline's pair token stream matches the legacy engine's, including a non-ASCII and an empty side.
#[test]
fn pairs_smoke() {
    let Ok(oracle) = Tokenizer::from_file(MODELS[0].1) else {
        eprintln!("bpe pairs smoke: skip -- {} not found", MODELS[0].1);
        return;
    };
    let pipeline = PipelineTokenizer::try_from(&oracle)
        .unwrap_or_else(|e| panic!("gpt2: pipeline construction failed: {e}"));

    let pairs = [
        ("hello world", "goodbye world"),
        ("café au lait", "日本語のテスト"),
        ("", "non-empty second"),
    ];
    for (a, b) in pairs {
        let expected = oracle.encode((a, b), false).unwrap();
        let got: Vec<u32> = pipeline
            .encode((a, b), false)
            .wait()
            .unwrap()
            .first()
            .unwrap()
            .ids()
            .iter()
            .map(|t| t.id())
            .collect();
        assert_eq!(
            expected.get_ids(),
            got.as_slice(),
            "pair id mismatch on ({a:?}, {b:?})"
        );
    }
}
