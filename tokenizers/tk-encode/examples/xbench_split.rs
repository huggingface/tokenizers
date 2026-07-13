//! Split sub-stage timing: the SIMD classify pass vs each FSM rule, in ns/B.
//! classify fills the tag stream; each fsm_* consumes precomputed tags → spans,
//! so the two are timed independently (no differencing).
//! Usage: xbench_split <corpus.json> [reps]
use atomsplit::classify::{classify, Atoms};
use atomsplit::fsm::{fsm_byte_level, fsm_cl100k, fsm_deepseek, fsm_o200k, Span};
use std::hint::black_box;
use std::time::Instant;

fn time(reps: usize, mut f: impl FnMut() -> usize) -> f64 {
    black_box(f());
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let docs: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let reps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
    let bytes: Vec<&[u8]> = docs.iter().map(|d| d.as_bytes()).collect();
    let n_bytes: usize = bytes.iter().map(|b| b.len()).sum();
    let maxlen = bytes.iter().map(|b| b.len()).max().unwrap_or(0);

    // precompute the tag stream for every doc (FSM input)
    let mut tags: Vec<Vec<u8>> = bytes
        .iter()
        .map(|b| {
            let mut t = vec![0u8; b.len()];
            classify::<Atoms>(b, &mut t);
            t
        })
        .collect();
    let mut spans: Vec<Span> = vec![(0, 0); maxlen + 8];

    let nsb = |sec: f64| sec * 1e9 / n_bytes as f64;
    let mbps = |sec: f64| n_bytes as f64 / sec / 1e6;

    let t_classify = time(reps, || {
        let mut n = 0;
        for (b, t) in bytes.iter().zip(tags.iter_mut()) {
            classify::<Atoms>(b, &mut t[..]);
            n += t.len();
        }
        n
    });
    println!("classify (SIMD tag pass)   {:6.3} ns/B   ({:.0} MB/s)", nsb(t_classify), mbps(t_classify));

    for (name, f) in [
        ("fsm_byte_level (gpt2)", fsm_byte_level as fn(&[u8], &[u8], &mut [Span]) -> usize),
        ("fsm_cl100k", fsm_cl100k),
        ("fsm_o200k", fsm_o200k),
        ("fsm_deepseek", fsm_deepseek),
    ] {
        let t = time(reps, || {
            let mut n = 0;
            for (b, tg) in bytes.iter().zip(tags.iter()) {
                n += f(b, tg, &mut spans);
            }
            n
        });
        println!("{:26} {:6.3} ns/B   ({:.0} MB/s)   [FSM only, tags precomputed]", name, nsb(t), mbps(t));
    }
    println!("n_bytes {}", n_bytes);
}
