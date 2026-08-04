//! Pure-Rust ceiling for the node-binding comparison: same models, same corpora,
//! same call shape as `bindings/node` `PipelineTokenizer.encode` — so the delta
//! against the node numbers is exactly the napi boundary.
//!
//! Usage: node_ref <tokenizer.json> <corpus.txt>
use std::hint::black_box;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const SHORT: &str =
    "Summarize the following article in three bullet points, then rate its clarity from 1 to 10.";
const MIN_SECS: f64 = 2.0;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("tokenizer.json");
    let corpus = args.next().expect("corpus.txt");

    let tok = Tokenizer::from_file(&model).expect("load");
    let pipe = PipelineTokenizer::try_from(&tok).expect("pipeline");
    let text = std::fs::read_to_string(&corpus).expect("corpus");
    let bytes = text.len();

    // Throughput: best pass over the whole corpus, warm cache.
    black_box(pipe.encode(&text, true).unwrap());
    let (mut best, mut total) = (0f64, 0f64);
    while total < MIN_SECS {
        let t = Instant::now();
        black_box(pipe.encode(&text, true).unwrap());
        let s = t.elapsed().as_secs_f64();
        total += s;
        best = best.max(bytes as f64 / 1e6 / s);
    }

    // Per-call latency on the short prompt.
    for _ in 0..1000 {
        black_box(pipe.encode(SHORT, true).unwrap());
    }
    let n = 200_000;
    let mut lat = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..n {
            black_box(pipe.encode(SHORT, true).unwrap());
        }
        lat = lat.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);
    }

    println!("{best:.1} MB/s   {lat:.0} ns/call");
}
