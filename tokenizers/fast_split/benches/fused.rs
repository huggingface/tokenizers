//! Two-pass (SIMD `classify::<Atoms>` → fsm) vs FUSED one-pass scalar, per pre-tokenizer.
//!
//! The UnicodeScripts lesson: when classification is a cheap lookup and the fsm can't SIMD-skip, the
//! tag buffer is pure overhead and a fused scalar wins. This measures where that line falls for every
//! atom-engine pre-tokenizer. Fused = the SAME fsm logic (const masks, like the two-pass) with
//! `classify_char` inlined per char — one pass, no tag buffer. Both sides are called DIRECTLY (not via
//! fn pointers) so the timing closure inlines them equally; a fn-pointer table skews fused ~2× slow by
//! blocking `classify_char` from inlining into the fused loop.
//!
//! speedup = fused_time / twopass_time  →  >1 means the two-pass SIMD engine wins; <1 means fuse it.
//!
//! Run: cargo bench --bench fused
use fast_split::atom_tables::ATOM_TABLES;
use fast_split::classify::{char_len, classify, in_mask, mask, Atom, Atoms};
use fast_split::fsm::{fsm_class_runs, fsm_cl100k_simd, fsm_split, Behavior, Span};
#[cfg(target_arch = "aarch64")]
use fast_split::fsm::whitespace_split_simd;
#[cfg(not(target_arch = "aarch64"))]
use fast_split::fsm::whitespace_split_scalar;
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

#[inline(always)]
fn atom_at(text: &[u8], i: usize) -> u8 {
    ATOM_TABLES.classify_char(text, i)
}
#[inline(always)]
fn run_end_fused(text: &[u8], mut i: usize, end: usize, m: u16) -> usize {
    while i < end && in_mask(atom_at(text, i), m) {
        i += char_len(text[i]);
    }
    i
}

// ── fused generic split (mirror of fsm_split, classify inlined, no tag buffer) ────────────────────
#[inline(always)]
fn fused_split<const DELIM: u16, const BEHAVIOR: u8>(text: &[u8], out: &mut Vec<Span>) {
    out.clear();
    let n = text.len();
    let mut segs: Vec<(u32, u32, bool)> = Vec::new();
    let mut run_start = 0usize;
    let mut i = 0usize;
    while i < n {
        let s = i;
        i += char_len(text[s]);
        if in_mask(atom_at(text, s), DELIM) {
            if run_start != s {
                segs.push((run_start as u32, s as u32, false));
            }
            segs.push((s as u32, i as u32, true));
            run_start = i;
        }
    }
    if run_start != n {
        segs.push((run_start as u32, n as u32, false));
    }
    match BEHAVIOR {
        b if b == Behavior::Removed as u8 => {
            for &(s, e, m) in &segs {
                if !m {
                    out.push((s, e));
                }
            }
        }
        b if b == Behavior::Isolated as u8 => {
            for &(s, e, _) in &segs {
                out.push((s, e));
            }
        }
        _ /* Contiguous */ => {
            let mut pm = false;
            for &(s, e, m) in &segs {
                if m == pm && !out.is_empty() {
                    out.last_mut().unwrap().1 = e;
                } else {
                    out.push((s, e));
                }
                pm = m;
            }
        }
    }
}

// ── fused class-runs (mirror of fsm_class_runs) ──────────────────────────────────────────────────
#[inline(always)]
fn fused_class_runs<const DROP: u16, const ISOLATE: u16, const KEEP_SPLIT: u16>(
    text: &[u8],
    out: &mut Vec<Span>,
) {
    out.clear();
    let n = text.len();
    let mut open: Option<(usize, u8)> = None;
    let mut i = 0usize;
    while i < n {
        let s = i;
        i += char_len(text[s]);
        let a = atom_at(text, s);
        let class = if in_mask(a, DROP) {
            0
        } else if in_mask(a, ISOLATE) {
            1
        } else if in_mask(a, KEEP_SPLIT) {
            2
        } else {
            3
        };
        match class {
            0 => {
                if let Some((st, _)) = open.take() {
                    out.push((st as u32, s as u32));
                }
            }
            1 => {
                if let Some((st, _)) = open.take() {
                    out.push((st as u32, s as u32));
                }
                out.push((s as u32, i as u32));
            }
            c => match open {
                Some((_, oc)) if oc == c => {}
                Some((st, _)) => {
                    out.push((st as u32, s as u32));
                    open = Some((s, c));
                }
                None => open = Some((s, c)),
            },
        }
    }
    if let Some((st, _)) = open {
        out.push((st as u32, n as u32));
    }
}

// ── fused cl100k (mirror of cl100k::<false>, classify inlined, scalar run-ends) ───────────────────
#[inline(always)]
fn fused_cl100k(text: &[u8], out: &mut Vec<Span>) {
    out.clear();
    const LET: u8 = Atom::Letter as u8;
    const NW: u8 = Atom::NumWord as u8;
    const NO: u8 = Atom::NumOther as u8;
    const NLN: u8 = Atom::Newline as u8;
    const SPC: u8 = Atom::Space as u8;
    const WSO: u8 = Atom::WsOther as u8;
    const MRK: u8 = Atom::Mark as u8;
    const CON: u8 = Atom::Connector as u8;
    const PUN: u8 = Atom::Punct as u8;
    const APO: u8 = Atom::Apostrophe as u8;
    const SYM: u8 = Atom::SymOther as u8;
    const NMO: u8 = Atom::NumericOther as u8;
    const CTL: u8 = Atom::Control as u8;
    let end = text.len();
    let letters = |a: usize| run_end_fused(text, a, end, mask::LETTER);
    let other = |sp0: usize| -> usize {
        let mut p = run_end_fused(text, sp0, end, mask::NOT_WS_L_N);
        if p > sp0 {
            while p < end && atom_at(text, p) == NLN {
                p += char_len(text[p]);
            }
        }
        p
    };
    let ws = |i: usize| -> usize {
        let re = run_end_fused(text, i, end, mask::WS);
        if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
            i + r + 1
        } else if re == end {
            re
        } else {
            let mut last = re - 1;
            while last > i && text[last] & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > i {
                last
            } else {
                re
            }
        }
    };
    let mut i = 0;
    while i < end {
        let start = i;
        let b = text[i];
        match atom_at(text, i) {
            LET => i = letters(i),
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < 3 && in_mask(atom_at(text, p), mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            SPC => {
                let a = i + 1;
                i = if a < end && atom_at(text, a) == LET {
                    letters(a)
                } else {
                    let p = other(a);
                    if p > a { p } else { ws(i) }
                };
            }
            WSO => {
                let a = i + char_len(b);
                i = if a < end && atom_at(text, a) == LET { letters(a) } else { ws(i) };
            }
            NLN => i = ws(i),
            APO => {
                let mut adv = 0;
                if i + 1 < end && text[i + 1] < 0x80 {
                    let lc = text[i + 1] | 0x20;
                    adv = match lc {
                        b's' | b't' | b'm' | b'd' => 2,
                        b'r' | b'v' | b'l' if i + 2 < end && text[i + 2] < 0x80 => {
                            let l2 = text[i + 2] | 0x20;
                            usize::from(
                                (lc == b'r' && l2 == b'e')
                                    || (lc == b'v' && l2 == b'e')
                                    || (lc == b'l' && l2 == b'l'),
                            ) * 3
                        }
                        _ => 0,
                    };
                }
                i = if adv > 0 {
                    i + adv
                } else {
                    let a = i + 1;
                    if a < end && atom_at(text, a) == LET { letters(a) } else { other(i) }
                };
            }
            MRK | CON | PUN | SYM | NMO | CTL => {
                let a = i + char_len(b);
                i = if a < end && atom_at(text, a) == LET { letters(a) } else { other(i) };
            }
            _ => i += char_len(b),
        }
        out.push((start as u32, i as u32));
    }
}

// ── two-pass wrappers (SIMD classify + fsm), #[inline(always)] so the timing closure inlines them ──
#[inline(always)]
fn tp_wssplit(text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
    out.clear();
    classify::<Atoms>(text, tags);
    #[cfg(target_arch = "aarch64")]
    whitespace_split_simd(text, tags, out);
    #[cfg(not(target_arch = "aarch64"))]
    whitespace_split_scalar(text, tags, out);
}
#[inline(always)]
fn tp_punct(text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
    out.clear();
    classify::<Atoms>(text, tags);
    fsm_split::<{ mask::PUNCT }, { Behavior::Isolated as u8 }>(text, tags, out);
}
#[inline(always)]
fn tp_digits(text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
    out.clear();
    classify::<Atoms>(text, tags);
    fsm_split::<{ mask::NUMERIC }, { Behavior::Contiguous as u8 }>(text, tags, out);
}
#[inline(always)]
fn tp_ws(text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
    out.clear();
    classify::<Atoms>(text, tags);
    fsm_class_runs::<{ mask::WS }, 0, { mask::WORD }>(text, tags, out);
}
#[inline(always)]
fn tp_bert(text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
    out.clear();
    classify::<Atoms>(text, tags);
    fsm_class_runs::<{ mask::WS }, { mask::PUNCT }, 0>(text, tags, out);
}
#[inline(always)]
fn tp_cl100k(text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
    out.clear();
    classify::<Atoms>(text, tags);
    fsm_cl100k_simd(text, tags, out);
}

#[inline(always)]
fn fu_wssplit(text: &[u8], out: &mut Vec<Span>) {
    fused_split::<{ mask::WS }, { Behavior::Removed as u8 }>(text, out)
}
#[inline(always)]
fn fu_punct(text: &[u8], out: &mut Vec<Span>) {
    fused_split::<{ mask::PUNCT }, { Behavior::Isolated as u8 }>(text, out)
}
#[inline(always)]
fn fu_digits(text: &[u8], out: &mut Vec<Span>) {
    fused_split::<{ mask::NUMERIC }, { Behavior::Contiguous as u8 }>(text, out)
}
#[inline(always)]
fn fu_ws(text: &[u8], out: &mut Vec<Span>) {
    fused_class_runs::<{ mask::WS }, 0, { mask::WORD }>(text, out)
}
#[inline(always)]
fn fu_bert(text: &[u8], out: &mut Vec<Span>) {
    fused_class_runs::<{ mask::WS }, { mask::PUNCT }, 0>(text, out)
}
#[inline(always)]
fn fu_cl100k(text: &[u8], out: &mut Vec<Span>) {
    fused_cl100k(text, out)
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

// Direct (inlinable) calls — $tp/$fu are fn *identifiers*, not pointers, so the closure inlines the
// whole chain (wrapper → fsm/fused → classify_char). fn pointers would block that and skew fused slow.
macro_rules! bench {
    ($corpora:expr, $name:expr, $tp:ident, $fu:ident) => {{
        println!(
            "\n== {} ==\n{:<10} {:>7}  {:>10} {:>10} | {:>7} {:>6}",
            $name, "lang", "bytes", "2pass", "fused", "speedup", "parity"
        );
        for (label, corpus) in $corpora {
            let text = corpus.as_bytes();
            let n = text.len();
            let iters = (4_000_000 / n).clamp(3, 150) as u32;
            let mut tags = vec![0u8; n];
            let (mut a, mut b) = (Vec::new(), Vec::new());
            $tp(text, &mut tags, &mut a);
            $fu(text, &mut b);
            let parity = a == b;
            let t2 = ns_per_byte(n, iters, || {
                $tp(text, &mut tags, &mut a);
                a.len()
            });
            let tf = ns_per_byte(n, iters, || {
                $fu(text, &mut b);
                b.len()
            });
            println!(
                "{:<10} {:>7}  {:>10.3} {:>10.3} | {:>6.2}x {:>6}",
                label, n, t2, tf, tf / t2, if parity { "✓" } else { "✗" }
            );
        }
    }};
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let mut corpora: Vec<(&str, String)> = Vec::new();
    for (label, rel) in CORPORA {
        if let Ok(s) = std::fs::read_to_string(format!("{manifest}/{rel}")) {
            if !s.trim().is_empty() {
                let mut c = s.len().min(180_000);
                while c > 0 && !s.is_char_boundary(c) {
                    c -= 1;
                }
                corpora.push((label, s[..c].to_string()));
            }
        }
    }

    bench!(&corpora, "WhitespaceSplit", tp_wssplit, fu_wssplit);
    bench!(&corpora, "Punctuation", tp_punct, fu_punct);
    bench!(&corpora, "Digits", tp_digits, fu_digits);
    bench!(&corpora, "Whitespace \\w", tp_ws, fu_ws);
    bench!(&corpora, "Bert", tp_bert, fu_bert);
    bench!(&corpora, "Cl100k", tp_cl100k, fu_cl100k);

    println!(
        "\n(ns/byte, lower better. 2pass = SIMD classify::<Atoms> + fsm (classify timed);\n \
         fused = one scalar pass, classify_char inlined, no tag buffer.\n \
         speedup = fused/2pass: >1 → two-pass SIMD wins (keep it); <1 → fuse it.)"
    );
}
