//! Throwaway cross-engine harness: single-thread throughput of the Rust
//! `PipelineTokenizer` on one tokenizer.json over one JSON corpus (list[str]).
//! Prints one JSON line: {"ok":bool,"mbps":..,"n_bytes":..,"n_docs":..}.
//! Usage: xbench_pipe <tokenizer.json> <corpus.json> [reps]
use rayon::prelude::*;
use std::convert::TryFrom;
use std::hint::black_box;
use std::time::Instant;
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (model, corpus) = (&a[1], &a[2]);
    let threads: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let docs: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(corpus).unwrap()).unwrap();
    let n_bytes: usize = docs.iter().map(|d| d.len()).sum();

    let build = (|| -> Result<PipelineTokenizer, String> {
        let tok = Tokenizer::from_file(model).map_err(|e| e.to_string())?;
        PipelineTokenizer::try_from(&tok).map_err(|e| e.to_string())
    })();
    let pipe = match build {
        Ok(p) => p,
        Err(e) => {
            println!("{{\"ok\": false, \"err\": {:?}}}", e);
            return;
        }
    };

    // N-thread rayon pool; each doc gets thread-local buffers. Same method for
    // every thread count (incl. 1) so the scaling curve is apples-to-apples.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let run = || {
        pool.install(|| {
            docs.par_iter()
                .map(|d| {
                    let mut out = Vec::new();
                    let mut pre = Vec::new();
                    pipe.encode_generic::<{ PipelineTokenizer::STAGE_MODEL }>(d, &mut out, &mut pre)
                        .unwrap();
                    out.len()
                })
                .sum::<usize>()
        })
    };
    // XBENCH_COLD: one timed pass, NO warm-up — the FlatCache starts empty and
    // fills during the pass, so only within-document repeats hit (realistic
    // "encode this doc once"). Default: warm-up + min-of-reps (steady state).
    let cold = std::env::var("XBENCH_COLD").is_ok();
    let (n_tokens, best) = if cold {
        let t = Instant::now();
        let n = run();
        (n, t.elapsed().as_secs_f64())
    } else {
        let n = run(); // warm
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            black_box(run());
            best = best.min(t.elapsed().as_secs_f64());
        }
        (n, best)
    };
    println!(
        "{{\"ok\": true, \"mbps\": {:.4}, \"n_bytes\": {}, \"n_docs\": {}, \"n_tokens\": {}}}",
        n_bytes as f64 / best / 1e6,
        n_bytes,
        docs.len(),
        n_tokens
    );
}
