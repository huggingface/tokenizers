//! o200k (GPT-4o) pretokenization: my single-pass `fsm_o200k` vs the REAL pretokenizer — the o200k
//! regex applied as ONE Isolated `Split` (onig, backtracking, so the `(?!\S)` lookahead + case-aware
//! `[\p{Lu}\p{Lt}…]*[\p{Ll}…]+` alts behave exactly as HF ships them). No o200k fixture exists, so onig
//! IS the oracle. Byte-exactness gate (✓/✗) + per-language timing on big real text.
//!
//! Run: cargo bench --bench o200k
use atomsplit::classify::classify;
use atomsplit::fsm::{Span, fsm_o200k};
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

// The canonical o200k / GPT-4o pre-tokenization regex (matches tk-encode's `O200K` const).
const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hebrew", "benches/data/he.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

// One Isolated Split of the whole text by `re`: emit gaps + matches (all pieces).
fn o200k_ref(text: &str, re: &Regex) -> Vec<Span> {
    let mut out = Vec::new();
    let mut prev = 0usize;
    for (ms, me) in re.find_iter(text) {
        if ms > prev {
            out.push(Span::new(prev as u32, ms as u32));
        }
        out.push(Span::new(ms as u32, me as u32));
        prev = me;
    }
    if prev < text.len() {
        out.push(Span::new(prev as u32, text.len() as u32));
    }
    out
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

fn report_diff(corpus: &str, ours: &[Span], reference: &[Span]) -> &'static str {
    if ours == reference {
        return "✓";
    }
    let mut k = 0;
    while k < ours.len() && k < reference.len() && ours[k] == reference[k] {
        k += 1;
    }
    let (os, oe) = ours
        .get(k)
        .map_or((0, 0), |sp| (sp.start as usize, sp.end as usize));
    let (rs, re) = reference
        .get(k)
        .map_or((0, 0), |sp| (sp.start as usize, sp.end as usize));
    eprintln!(
        "  DIVERGE @tok {k}/{} (ref {}): ours[{os}..{oe}]={:?} ref[{rs}..{re}]={:?}",
        ours.len(),
        reference.len(),
        &corpus[os..oe.min(corpus.len())],
        &corpus[rs..re.min(corpus.len())],
    );
    "✗"
}

fn main() {
    let re = Regex::new(O200K).unwrap();
    let manifest = env!("CARGO_MANIFEST_DIR");

    println!(
        "{:<10} {:>7} {:>5}  {:>8} {:>8} | {:>8} | {:>7} {:>4}",
        "lang", "bytes", "b/tok", "clsSIMD", "fsmScal", "onig", "vsRef", "parity"
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

        // parity: fsm_o200k == the Isolated o200k Split
        let reference = o200k_ref(corpus, &re);
        let mut tags = vec![0u8; n];
        classify(text, &mut tags);
        let mut buf = vec![Span::default(); n + 1];
        let k = fsm_o200k(text, &tags, &mut buf);
        let parity = report_diff(corpus, &buf[..k], &reference);
        let btok = n as f64 / k.max(1) as f64;

        // timing
        let cls_simd = ns_per_byte(n, iters, || {
            classify(text, &mut tags);
            tags[n / 2] as usize
        });
        classify(text, &mut tags);
        let fsm_scal = ns_per_byte(n, iters, || fsm_o200k(text, &tags, &mut buf));
        let onig_ns = ns_per_byte(n, iters, || o200k_ref(corpus, &re).len());

        let pipe = cls_simd + fsm_scal;
        println!(
            "{label:<10} {n:>7} {btok:>5.1}  {cls_simd:>8.3} {fsm_scal:>8.3} | {onig_ns:>8.2} | {:>6.1}x {parity:>5}",
            onig_ns / pipe
        );
    }
    println!(
        "\n(ns/byte, lower better; onig = the Isolated o200k Split reference. parity: fsm_o200k == reference.)"
    );
}
