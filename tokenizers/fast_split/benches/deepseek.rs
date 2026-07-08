//! deepseek-v3 pretokenization: my single-pass `fsm_deepseek` vs the REAL pretokenizer — the
//! `Sequence` of three Isolated `Split`s (`\p{N}{1,3}` → CJK-range → big regex) composed with onig,
//! exactly as HF applies them (each split runs on the previous split's pieces, so lookaheads see
//! piece boundaries). Byte-exactness gate (✓/✗) + per-language timing on big real text.
//!
//! Run: cargo bench --bench deepseek
use fast_split::classify::{Atoms, classify, classify_scalar};
use fast_split::fsm::{Span, fsm_deepseek};
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

const P_NUM: &str = r"\p{N}{1,3}";
const P_CJK: &str = r"[一-龥぀-ゟ゠-ヿ]+";
// big regex (Split-3). r##"…"## because the pattern contains " and #.
const P_BIG: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

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

// One Isolated split of text[s..e] by `re`: emit gaps + matches (all pieces), absolute offsets.
fn split_iso(text: &str, s: usize, e: usize, re: &Regex, out: &mut Vec<(usize, usize)>) {
    let sub = &text[s..e];
    let mut prev = 0usize;
    for (ms, me) in re.find_iter(sub) {
        if ms > prev {
            out.push((s + prev, s + ms));
        }
        out.push((s + ms, s + me));
        prev = me;
    }
    if prev < sub.len() {
        out.push((s + prev, e));
    }
}

// The reference: the 3-split Sequence composed exactly as HF applies it.
fn deepseek_ref(text: &str, re_num: &Regex, re_cjk: &Regex, re_big: &Regex) -> Vec<Span> {
    let mut p1 = Vec::new();
    split_iso(text, 0, text.len(), re_num, &mut p1);
    let mut p2 = Vec::new();
    for (s, e) in p1 {
        split_iso(text, s, e, re_cjk, &mut p2);
    }
    let mut p3 = Vec::new();
    for (s, e) in p2 {
        split_iso(text, s, e, re_big, &mut p3);
    }
    p3.into_iter().map(|(s, e)| (s as u32, e as u32)).collect()
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
    let ctx = |lo: usize, hi: usize| {
        let (a, b) = (lo.saturating_sub(6), (hi + 6).min(corpus.len()));
        let (mut a, mut b) = (a, b);
        while !corpus.is_char_boundary(a) { a -= 1; }
        while !corpus.is_char_boundary(b) { b += 1; }
        corpus[a..b].escape_debug().to_string()
    };
    let (os, oe) = (ours[k].0 as usize, ours[k].1 as usize);
    let (rs, re) = (reference[k].0 as usize, reference[k].1 as usize);
    eprintln!(
        "  DIVERGE @tok {k}: ours[{os}..{oe}]={:?} ref[{rs}..{re}]={:?}  ctx={:?}",
        &corpus[os..oe],
        &corpus[rs..re],
        ctx(os.min(rs), oe.max(re))
    );
    "✗"
}

fn main() {
    let (rn, rc, rb) = (
        Regex::new(P_NUM).unwrap(),
        Regex::new(P_CJK).unwrap(),
        Regex::new(P_BIG).unwrap(),
    );
    let manifest = env!("CARGO_MANIFEST_DIR");

    println!(
        "{:<10} {:>7} {:>5}  {:>8} {:>8} | {:>8} | {:>7} {:>4}",
        "lang", "bytes", "b/tok", "clsSIMD", "fsmScal", "onig×3", "vsRef", "parity"
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

        // parity: fsm_deepseek == composed 3-split Sequence
        let reference = deepseek_ref(corpus, &rn, &rc, &rb);
        let mut tags = vec![0u8; n];
        classify::<Atoms>(text, &mut tags);
        let mut ours = Vec::new();
        fsm_deepseek(text, &tags, &mut ours);
        let parity = report_diff(corpus, &ours, &reference);
        let btok = n as f64 / ours.len().max(1) as f64;

        // timing
        let mut buf = Vec::with_capacity(ours.len());
        let cls_simd = ns_per_byte(n, iters, || {
            classify::<Atoms>(text, &mut tags);
            tags[n / 2] as usize
        });
        let mut tsc = vec![0u8; n];
        let _ = classify_scalar::<Atoms>; // (scalar classify measured in the cl100k bench; skip here)
        let _ = &mut tsc;
        classify::<Atoms>(text, &mut tags);
        let fsm_scal = ns_per_byte(n, iters, || {
            buf.clear();
            fsm_deepseek(text, &tags, &mut buf);
            buf.len()
        });
        let onig_ns = ns_per_byte(n, iters, || deepseek_ref(corpus, &rn, &rc, &rb).len());

        let pipe = cls_simd + fsm_scal;
        println!(
            "{label:<10} {n:>7} {btok:>5.1}  {cls_simd:>8.3} {fsm_scal:>8.3} | {onig_ns:>8.2} | {:>6.1}x {parity:>5}",
            onig_ns / pipe
        );
    }
    println!("\n(ns/byte, lower better; onig×3 = the composed Sequence reference. parity: fsm_deepseek == reference.)");
}
