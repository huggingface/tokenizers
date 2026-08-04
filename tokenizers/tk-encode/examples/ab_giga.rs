//! Our side of the gigatoken A/B: same protocol as the gigatoken `giga_corpus` bench --
//! one buffer of AB_MB MB built by repeating the corpus, AB_PASSES single-thread encodes,
//! pass 0 cold (fresh tokenizer, empty cache) and the rest warm. Prints the token count so
//! the comparison can be confirmed apples-to-apples.
//!
//! Best-of-N over the warm passes: this box swings ~20% run to run, so a single pass
//! cannot resolve a 10% difference.
//!
//! The **output buffer is reserved once and reused** across passes, because that is what
//! gigatoken's harness does (`Vec::with_capacity(len/4)` then `clear()` per pass). Returning
//! a fresh `Vec` per encode instead -- what `encode_one` does -- costs an allocation plus a
//! first-touch of the whole token array every pass, and that scales with TOKENS, not bytes:
//! chinese emits 2.9 M tokens for a 4 MB input against english's 0.8 M, so measuring it that
//! way quietly penalises exactly the token-dense corpora. Use `encode_generic`, which writes
//! into a caller-owned buffer and is otherwise the same full pipeline `encode_one` runs.
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::{Model, PipelineTokenizer};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data");
    let target = env_usize("AB_MB", 4) * 1024 * 1024;
    let passes = env_usize("AB_PASSES", 3).max(2);

    println!(
        "single-thread full encode, {} MB buffer, {passes} passes\n",
        target >> 20
    );
    println!(
        "  {:<24} {:<9} {:>10} {:>9} {:>9}",
        "model", "corpus", "tokens", "cold", "warm"
    );
    for model in [
        "gpt2.json",
        "llama-3-tokenizer.json",
        "deepseek-v4.json",
    ] {
        let path = root.join(model);
        if !path.exists() {
            continue;
        }
        for corpus in ["english", "code", "chinese", "korean", "thai", "hindi", "arabic", "greek", "russian", "dense"] {
            let Ok(text) =
                std::fs::read_to_string(root.join("corpora").join(format!("{corpus}.txt")))
            else {
                continue;
            };
            let one = text.repeat(target.div_ceil(text.len()));
            let mb = one.len() as f64 / (1024.0 * 1024.0);

            // Fresh tokenizer per corpus so pass 0 really is a cold cache.
            let legacy = Tokenizer::from_file(&path).expect("load");
            let Ok(pipe) = PipelineTokenizer::try_from(&legacy) else {
                continue;
            };

            let mut cold = 0.0;
            let mut warm: f64 = 0.0;
            let mut tokens = 0;
            // Reserved once, cleared per pass -- exactly gigatoken's harness.
            let mut out = Vec::with_capacity(one.len() / 4 + 16);
            let mut pre_tokens = Vec::new();
            let mut scratch = pipe.get_model().init_scratch();
            for pass in 0..passes {
                out.clear();
                let start = Instant::now();
                pipe.encode_generic::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(
                    &one,
                    false,
                    &mut pre_tokens,
                    &mut scratch,
                    &mut out,
                )
                .unwrap();
                let secs = start.elapsed().as_secs_f64();
                tokens = out.len();
                black_box(&out);
                let rate = mb / secs;
                if pass == 0 {
                    cold = rate;
                } else {
                    warm = warm.max(rate);
                }
            }
            println!(
                "  {model:<24} {corpus:<9} {tokens:>10} {cold:>9.0} {warm:>9.0}"
            );
        }
    }
}
