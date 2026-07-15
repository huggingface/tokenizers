//! GPT pre-tokenization on BIG real text (Wikipedia / big.txt), per language, for every regex family at
//! once — gpt2 · cl100k · o200k · deepseek. Per (engine, corpus): classify (SIMD vs scalar) + fsm in
//! ns/byte, and the full pipeline (SIMD classify + scalar fsm) vs onig AND fancy-regex, with a
//! byte-exactness gate (✓/✗) against onig. The reference regex(es) are the canonical ones in
//! [`atomsplit::regexes`] — the same patterns the FSMs implement and the pipeline recognizes, so nothing
//! can drift. Single-regex engines have a 1-element chain; deepseek composes 3 `Isolated` Splits.
//!
//! Data: `../data/big.txt` (English) + `../data/unigram_wagahaiwa_nekodearu.txt` (Japanese) ship with
//! the repo; the rest via `benches/data/fetch.py` (gitignored). Missing files are skipped.
//!
//! Run: cargo bench --bench regex
use atomsplit::classify::{classify, classify_scalar};
use atomsplit::fsm::{Span, fsm_byte_level, fsm_cl100k, fsm_deepseek, fsm_o200k};
use atomsplit::regexes;
use fancy_regex::Regex as Fancy;
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

type Fsm = fn(&[u8], &[u8], &mut [Span]) -> usize;

// (name, native fsm, reference regex chain). The chain is applied `Isolated`, each regex splitting the
// previous pieces — one element for gpt2/cl100k/o200k, three for deepseek.
const ENGINES: &[(&str, Fsm, &[&str])] = &[
    ("gpt2", fsm_byte_level as Fsm, &[regexes::GPT2]),
    ("cl100k", fsm_cl100k as Fsm, &[regexes::CL100K]),
    ("o200k", fsm_o200k as Fsm, &[regexes::O200K]),
    ("deepseek", fsm_deepseek as Fsm, regexes::DEEPSEEK),
];

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Hebrew", "benches/data/he.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

/// The composed `Isolated` split under `res` (each regex splits the previous pieces into gaps + matches)
/// as byte-offset pieces into `text`. For a single full-coverage regex this is just its match list.
fn onig_pieces(res: &[Regex], text: &str) -> Vec<(usize, usize)> {
    let mut pieces = vec![(0usize, text.len())];
    for re in res {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for (s, e) in pieces.drain(..) {
            let sub = &text[s..e];
            let mut prev = 0usize;
            for (ms, me) in re.find_iter(sub) {
                if ms > prev {
                    next.push((s + prev, s + ms));
                }
                next.push((s + ms, s + me));
                prev = me;
            }
            if prev < sub.len() {
                next.push((s + prev, e));
            }
        }
        pieces = next;
    }
    pieces
}

/// Same composition under fancy-regex; `find_iter` yields `Result<Match>` (a match error ends the pass).
fn fancy_pieces(res: &[Fancy], text: &str) -> Vec<(usize, usize)> {
    let mut pieces = vec![(0usize, text.len())];
    for re in res {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for (s, e) in pieces.drain(..) {
            let sub = &text[s..e];
            let mut prev = 0usize;
            for m in re.find_iter(sub) {
                let Ok(m) = m else { break };
                let (ms, me) = (m.start(), m.end());
                if ms > prev {
                    next.push((s + prev, s + ms));
                }
                next.push((s + ms, s + me));
                prev = me;
            }
            if prev < sub.len() {
                next.push((s + prev, e));
            }
        }
        pieces = next;
    }
    pieces
}

// MIN over TRIALS timed loops — the fastest trial had the least CPU contention, so it's the truest
// estimate and robust to thermal throttling / background load (which only ever make a trial slower).
fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    const TRIALS: u32 = 7;
    for _ in 0..3 {
        black_box(f()); // warm
    }
    let mut best = f64::INFINITY;
    for _ in 0..TRIALS {
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
    let manifest = env!("CARGO_MANIFEST_DIR");
    println!(
        "{:<9} {:<10} {:>7} {:>5}  {:>8} {:>8} | {:>8} | {:>8} {:>8} | {:>7} {:>7}",
        "engine", "lang", "bytes", "b/tok", "clsSIMD", "clsScal", "fsm", "onig", "fancy", "vsOnig",
        "vsFncy"
    );
    for &(ename, fsm, rxs) in ENGINES {
        let onig: Vec<Regex> = rxs.iter().map(|r| Regex::new(r).expect(ename)).collect();
        let fancy: Vec<Fancy> = rxs.iter().map(|r| Fancy::new(r).expect(ename)).collect();
        for (label, rel) in CORPORA {
            let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
                Ok(s) if !s.trim().is_empty() => s,
                _ => {
                    println!("{ename:<9} {label:<10}  (skipped — {rel} missing; run benches/data/fetch.py)");
                    continue;
                }
            };
            // UNIFORM cap (char boundary): every language the same byte size → equal cache behaviour, so
            // per-byte compute compares across scripts. 180 KB > L1, and all corpora have ≥180 KB.
            let mut c = raw.len().min(180_000);
            while c > 0 && !raw.is_char_boundary(c) {
                c -= 1;
            }
            let corpus = &raw[..c];
            let text = corpus.as_bytes();
            let n = text.len();
            let iters = (4_000_000 / n).clamp(3, 150) as u32;

            let ref_spans: Vec<Span> = onig_pieces(&onig, corpus)
                .iter()
                .map(|&(s, e)| Span::new(s as u32, e as u32))
                .collect();
            let mut tags = vec![0u8; n];
            let mut tsc = vec![0u8; n];
            classify(text, &mut tags);
            let mut buf = vec![Span::default(); n + 1];
            let k = fsm(text, &tags, &mut buf);
            let ok = if buf[..k] == ref_spans[..] { "✓" } else { "✗" };
            let btok = n as f64 / k.max(1) as f64;

            let cls_simd = ns_per_byte(n, iters, || {
                classify(text, &mut tags);
                tags[n / 2] as usize
            });
            let cls_scal = ns_per_byte(n, iters, || {
                classify_scalar(text, &mut tsc);
                tsc[n / 2] as usize
            });
            classify(text, &mut tags);
            let fsm_ns = ns_per_byte(n, iters, || fsm(text, &tags, &mut buf));
            let onig_ns = ns_per_byte(n, iters, || onig_pieces(&onig, corpus).len());
            let fancy_ns = ns_per_byte(n, iters, || fancy_pieces(&fancy, corpus).len());

            let pipe = cls_simd + fsm_ns; // SIMD classify + scalar fsm — the full pipeline
            println!(
                "{ename:<9} {label:<10} {n:>7} {btok:>5.1}  {cls_simd:>8.3} {cls_scal:>8.3} | {fsm_ns:>8.3} | {onig_ns:>8.2} {fancy_ns:>8.2} | {:>6.1}x {:>6.1}x {ok}",
                onig_ns / pipe,
                fancy_ns / pipe
            );
        }
    }
    println!(
        "\n(ns/byte, lower better. pipeline = SIMD classify + scalar fsm; vs onig / vs fancy on the \
         composed Isolated split. ✓ = fsm == onig, byte-exact.)"
    );
}
