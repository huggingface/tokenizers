//! Scalar vs SIMD fsm for the class family: `class_runs_runend` (scalar run-end core) vs
//! `class_runs_into` (NEON/SIMD128 movemask boundary-extract + homogeneous-chunk early-out). classify is
//! timed separately. `spd = scalar/simd`: >1 → SIMD wins. (cl100k is scalar-only — see benches/cl100k.rs.)
//!
//! Run: cargo bench --bench class_runs
use atomsplit::classify::{classify, mask};
use atomsplit::fsm::{Span, class_runs_into, class_runs_runend};
use std::hint::black_box;
use std::time::Instant;

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

// scalar (run-end core) vs SIMD (class_runs_into) wrappers per recipe — #[inline(always)] so the timing
// closure inlines the whole chain (not a fn-pointer, which would skew the scalar path slow).
macro_rules! pair {
    ($sname:ident, $vname:ident, $d:expr, $i:expr, $a:expr) => {
        #[inline(always)]
        fn $sname(t: &[u8], tg: &[u8], o: &mut [Span]) -> usize {
            class_runs_runend::<$d, $i, $a>(t, tg, o)
        }
        #[inline(always)]
        fn $vname(t: &[u8], tg: &[u8], o: &mut [Span]) -> usize {
            class_runs_into::<$d, $i, $a>(t, tg, o)
        }
    };
}
pair!(s_wss, v_wss, { mask::WS }, 0, 0);
pair!(s_pun, v_pun, 0, { mask::PUNCT }, 0);
pair!(s_dig, v_dig, 0, 0, { mask::NUMERIC });
pair!(s_ws, v_ws, { mask::WS }, 0, { mask::WORD });
pair!(s_bert, v_bert, { mask::WS }, { mask::PUNCT }, 0);

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
    let manifest = env!("CARGO_MANIFEST_DIR");
    let mut corpora: Vec<(&str, String)> = Vec::new();
    for (label, rel) in CORPORA {
        let Ok(s) = std::fs::read_to_string(format!("{manifest}/{rel}")) else {
            continue;
        };
        if s.trim().is_empty() {
            continue;
        }
        let mut c = s.len().min(180_000);
        while c > 0 && !s.is_char_boundary(c) {
            c -= 1;
        }
        corpora.push((label, s[..c].to_string()));
    }

    macro_rules! compare {
        ($name:expr, $s:ident, $v:ident) => {{
            println!(
                "\n== {} ==\n{:<10} {:>7}  {:>8} {:>8} {:>8} | {:>6}",
                $name, "lang", "bytes", "classify", "scalar", "simd", "spd"
            );
            for (label, corpus) in &corpora {
                let text = corpus.as_bytes();
                let n = text.len();
                let iters = (4_000_000 / n).clamp(3, 150) as u32;
                let mut tags = vec![0u8; n];
                let mut buf = vec![(0u32, 0u32); n + 1];
                classify(text, &mut tags);
                // parity: scalar == simd
                let (ks, kv) = ($s(text, &tags, &mut buf), 0);
                let scalar_out: Vec<Span> = buf[..ks].to_vec();
                let _ = kv;
                let kv = $v(text, &tags, &mut buf);
                let parity = scalar_out == buf[..kv];
                let cls = ns_per_byte(n, iters, || {
                    classify(text, &mut tags);
                    n
                });
                classify(text, &mut tags);
                let sc = ns_per_byte(n, iters, || $s(text, &tags, &mut buf));
                let si = ns_per_byte(n, iters, || $v(text, &tags, &mut buf));
                println!(
                    "{label:<10} {n:>7}  {cls:>8.3} {sc:>8.3} {si:>8.3} | {:>5.2}x{}",
                    sc / si,
                    if parity { "" } else { "  PARITY✗" }
                );
            }
        }};
    }

    compare!("WhitespaceSplit", s_wss, v_wss);
    compare!("Punctuation", s_pun, v_pun);
    compare!("Digits", s_dig, v_dig);
    compare!("Whitespace \\w", s_ws, v_ws);
    compare!("Bert", s_bert, v_bert);
    println!(
        "\n(ns/byte, lower better. classify = SIMD classify; scalar = class_runs_runend (run-end\n \
         core); simd = class_runs_into (NEON/SIMD128 movemask + early-out). spd = scalar/simd, >1 → SIMD\n \
         wins. cl100k is scalar-only — its perf is in benches/cl100k.rs.)"
    );
}
