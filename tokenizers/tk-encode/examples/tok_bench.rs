//! Throughput of the `.tok` read path: best-of-5 per corpus, ids only, single thread.
//! Mirrors `/tmp/et_bench.cpp` so the numbers compare directly.

use std::time::Instant;

use tk_encode::pipeline::PipelineTokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: tok_bench <tokenizer.tok> <corpus.txt>...");

    let t0 = Instant::now();
    let file = tk_serialization::TokFile::open(&path).expect("open .tok");
    let tok = PipelineTokenizer::from_tok(file.bytes()).expect("load .tok");
    println!("load {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    for corpus in args {
        let Ok(text) = std::fs::read_to_string(&corpus) else { continue };
        if text.is_empty() {
            continue;
        }
        let mb = text.len() as f64 / 1e6;
        let (mut best, mut n_ids) = (0f64, 0usize);
        for _ in 0..5 {
            let s = Instant::now();
            let ids = tok.encode(text.as_str(), false).expect("encode");
            let secs = s.elapsed().as_secs_f64();
            n_ids = ids.len();
            best = best.max(mb / secs);
        }
        println!("{corpus:<26} {n_ids:>8} ids  {best:>7.1} MB/s");
    }
}
