//! GPT-2 / ByteLevel pretokenization: `fsm_byte_level` vs the GPT-2 regex (onig + fancy), byte-exact
//! gate (✓/✗) + per-language timing. The ByteLevel regex is applied WITHOUT its byte-mapping step
//! (that's a post-split byte→char remap; the split boundaries are what this checks).
//!
//! Run: cargo bench --bench byte_level
use fast_split::classify::{Atoms, classify};
use fast_split::fsm::{Span, fsm_byte_level};
use fancy_regex::Regex as Fancy;
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

// GPT-2 pattern. r##"…"## because it contains " and #.
const GPT2: &str =
    r##"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"##;

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

fn onig_spans(re: &Regex, text: &str) -> Vec<Span> {
    re.find_iter(text).map(|(a, b)| (a as u32, b as u32)).collect()
}

fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        let mut acc = 0usize;
        for _ in 0..iters {
            acc = acc.wrapping_add(f());
        }
        black_box(acc);
        best = best.min(t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64);
    }
    best
}

fn main() {
    let re = Regex::new(GPT2).expect("gpt2 onig");
    let fancy = Fancy::new(GPT2).expect("gpt2 fancy");
    let manifest = env!("CARGO_MANIFEST_DIR");

    println!(
        "{:<10} {:>7} {:>5}  {:>8} {:>8} | {:>8} {:>8} | {:>7} {:>7} {:>4}",
        "lang", "bytes", "b/tok", "clsSIMD", "fsmScal", "onig", "fancy", "vsOnig", "vsFncy", "ok"
    );

    for (label, rel) in CORPORA {
        let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                println!("{label:<10}  (skipped — {rel} missing)");
                continue;
            }
        };
        let mut c = raw.len().min(180_000);
        while c > 0 && !raw.is_char_boundary(c) {
            c -= 1;
        }
        let corpus = &raw[..c];
        let text = corpus.as_bytes();
        let n = text.len();
        let iters = (4_000_000 / n).clamp(3, 150) as u32;

        let reference = onig_spans(&re, corpus);
        let mut tags = vec![0u8; n];
        classify::<Atoms>(text, &mut tags);
        let mut sc = vec![(0u32, 0u32); n + 1];
        let k = fsm_byte_level(text, &tags, &mut sc);
        let ok = if sc[..k] == reference[..] { "✓" } else { "✗" };
        let btok = n as f64 / k.max(1) as f64;

        let cls = ns_per_byte(n, iters, || {
            classify::<Atoms>(text, &mut tags);
            tags[n / 2] as usize
        });
        classify::<Atoms>(text, &mut tags);
        let fsm = ns_per_byte(n, iters, || fsm_byte_level(text, &tags, &mut sc));
        let onig_ns = ns_per_byte(n, iters, || onig_spans(&re, corpus).len());
        let fancy_ns = ns_per_byte(n, iters, || fancy.find_iter(corpus).count());

        let pipe = cls + fsm;
        println!(
            "{label:<10} {n:>7} {btok:>5.1}  {cls:>8.3} {fsm:>8.3} | {onig_ns:>8.2} {fancy_ns:>8.2} | {:>6.1}x {:>6.1}x {ok}",
            onig_ns / pipe,
            fancy_ns / pipe
        );
    }
    println!("\n(ns/byte, lower better. ok = fsm_byte_level == GPT-2 onig regex.)");
}
