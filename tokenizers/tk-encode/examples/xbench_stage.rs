//! Stage breakdown of the pipeline encode: time encode_generic at each STAGE
//! (0 frame, 1 +normalize, 2 +split, 3 +merge/model); the delta between stages
//! is that stage's cost. Run with POC_NOCACHE=1 for the pure-compute breakdown.
//! Usage: xbench_stage <tokenizer.json> <corpus(.json|.txt)> [reps]
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
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let docs = chunks_from(&a[2]);
    let n_bytes: usize = docs.iter().map(|d| d.len()).sum();
    let tok = Tokenizer::from_file(&a[1]).unwrap();
    let pipe = PipelineTokenizer::try_from(&tok).unwrap();

    let mut out = Vec::new();
    let mut pre = Vec::new();
    let time_stage = |stage: u8, out: &mut Vec<_>, pre: &mut Vec<_>| -> f64 {
        let run = |out: &mut Vec<_>, pre: &mut Vec<_>| {
            let mut n = 0usize;
            for d in &docs {
                out.clear();
                pre.clear();
                match stage {
                    0 => pipe.encode_generic::<0>(d, out, pre).unwrap(),
                    1 => pipe.encode_generic::<1>(d, out, pre).unwrap(),
                    2 => pipe.encode_generic::<2>(d, out, pre).unwrap(),
                    _ => pipe.encode_generic::<3>(d, out, pre).unwrap(),
                }
                n += out.len();
            }
            n
        };
        black_box(run(out, pre));
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            black_box(run(out, pre));
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    let t: Vec<f64> = (0u8..=3).map(|s| time_stage(s, &mut out, &mut pre)).collect();
    let names = ["frame/scan", "normalize", "split", "merge/model"];
    let total = t[3];
    println!("stage breakdown ({} bytes, min of {} reps):", n_bytes, reps);
    let mut prev = 0.0;
    for (i, name) in names.iter().enumerate() {
        let dt = (t[i] - prev).max(0.0);
        println!(
            "  {:12}  {:7.2} ms  {:5.1}%   ({:.0} MB/s at this stage)",
            name,
            t[i] * 1e3,
            dt / total * 100.0,
            n_bytes as f64 / t[i] / 1e6
        );
        prev = t[i];
    }
    println!("  TOTAL         {:7.2} ms          ({:.0} MB/s)", total * 1e3, n_bytes as f64 / total / 1e6);
}
