//! Per-iteration pipeline throughput (no warm-up): iteration 1 is COLD (cache
//! starts empty, fills during the pass over distinct chunks), iterations 2+ are
//! WARM. Run with the FlatCache on (default) and off (POC_NOCACHE=1) to show
//! the cache never slows the cold pass and is far faster once warm.
//! Usage: xbench_iter <tokenizer.json> <corpus(.json list | .txt)> [iters]
use std::convert::TryFrom;
use std::hint::black_box;
use std::time::Instant;
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

fn chunks_from(path: &str) -> Vec<String> {
    let raw = std::fs::read_to_string(path).unwrap();
    if path.ends_with(".json") {
        return serde_json::from_str(&raw).unwrap();
    }
    // ~10 kB chunks on char boundaries — mirrors the CI fixture_bench regime.
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in raw.lines() {
        cur.push_str(line);
        cur.push('\n');
        if cur.len() >= 10_000 {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (model, corpus) = (&a[1], &a[2]);
    let iters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);

    let docs = chunks_from(corpus);
    let n_bytes: usize = docs.iter().map(|d| d.len()).sum();
    let tok = Tokenizer::from_file(model).unwrap();
    let pipe = PipelineTokenizer::try_from(&tok).unwrap();

    let mut out = Vec::new();
    let mut pre = Vec::new();
    let mut mbps = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        let mut n = 0usize;
        for d in &docs {
            out.clear();
            pre.clear();
            pipe.encode_generic::<{ PipelineTokenizer::STAGE_MODEL }>(d, &mut out, &mut pre)
                .unwrap();
            n += out.len();
        }
        black_box(n);
        mbps.push(n_bytes as f64 / t.elapsed().as_secs_f64() / 1e6);
    }
    let s: Vec<String> = mbps.iter().map(|m| format!("{m:.1}")).collect();
    // CACHE_STATS=1: report cumulative hit rate (run with iters=1 for the cold single-pass rate).
    let (hits, misses) = tk_encode::models::bpe::flat_cache_stats();
    let hr = if hits + misses > 0 {
        hits as f64 / (hits + misses) as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "{{\"n_docs\": {}, \"n_bytes\": {}, \"mbps_per_iter\": [{}], \"hits\": {}, \"misses\": {}, \"hit_rate\": {:.1}}}",
        docs.len(),
        n_bytes,
        s.join(", "),
        hits,
        misses,
        hr
    );
}
