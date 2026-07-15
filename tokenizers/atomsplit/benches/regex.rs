//! Pre-tokenization on BIG real text (Wikipedia / big.txt), per language, for every pre-tokenizer at
//! once — the GPT regex FSMs (gpt2 · cl100k · o200k · deepseek) and the class family (WhitespaceSplit ·
//! Whitespace · Digits · Punctuation · Bert). Per (engine, corpus): classify (SIMD vs scalar) + fsm in
//! ns/byte, and the full pipeline (SIMD classify + scalar fsm) vs FOUR reference engines — onig (C),
//! fancy-regex (pure Rust), logos (compile-time DFA lexer), pcre2 (C + JIT). GPT regexes come from
//! [`atomsplit::regexes`] (the canonical specs, byte-exact ✓/✗); class-family references are the
//! best-effort equivalent regex (approximate — see the ≈ marker). Single-regex engines have a 1-element
//! chain; deepseek composes 3 `Isolated` Splits.
//!
//! Data: `../data/big.txt` (English) + `../data/unigram_wagahaiwa_nekodearu.txt` (Japanese) ship with
//! the repo; the rest via `benches/data/fetch.py` (gitignored). Missing files are skipped.
//!
//! Run: cargo bench --bench regex
use atomsplit::classify::{classify, classify_scalar, mask};
use atomsplit::fsm::{Span, class_runs_into, fsm_byte_level, fsm_cl100k, fsm_deepseek, fsm_o200k};
use atomsplit::regexes;
use fancy_regex::Regex as Fancy;
use logos::Logos;
use onig::Regex;
use pcre2::bytes::Regex as Pcre2;
use std::hint::black_box;
use std::time::Instant;

type Fsm = fn(&[u8], &[u8], &mut [Span]) -> usize;

// logos DFA lexers approximating the GPT splits. logos has no look-ahead (`(?!\S)`) nor case-insensitive
// (`(?i:)`), so token boundaries differ slightly — this is a raw-throughput reference (like fancy), not a
// byte-exact oracle (that's onig). deepseek is a 3-split `Sequence`, not one grammar → no logos number.
#[derive(Logos)]
enum LGpt2 {
    #[regex(r"'s|'t|'re|'ve|'m|'ll|'d")]
    Contraction,
    #[regex(r" ?\p{L}+")]
    Word,
    #[regex(r" ?\p{N}+")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+")]
    Other,
    #[regex(r"\s+")]
    Space,
}

#[derive(Logos)]
enum LCl100k {
    // Contraction outranks Word: `[^\r\n\p{L}\p{N}]?\p{L}+` can also match `'s`, but the real regex tries
    // the contraction alternative first.
    #[regex(r"'s|'t|'re|'ve|'m|'ll|'d", priority = 5)]
    Contraction,
    #[regex(r"[^\r\n\p{L}\p{N}]?\p{L}+", priority = 4)]
    Word,
    #[regex(r"\p{N}\p{N}?\p{N}?")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n]*", priority = 2)]
    Other,
    #[regex(r"\s+")]
    Space,
}

#[derive(Logos)]
enum LO200k {
    #[regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+('s|'t|'re|'ve|'m|'ll|'d)?", priority = 6)]
    LettersA,
    #[regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*('s|'t|'re|'ve|'m|'ll|'d)?", priority = 5)]
    LettersB,
    #[regex(r"\p{N}\p{N}?\p{N}?")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n/]*", priority = 2)]
    Other,
    #[regex(r"\s+")]
    Space,
}

// class-family logos grammars. The whitespace-dropping ones use `skip` so logos never emits ws tokens.
#[derive(Logos)]
#[logos(skip r"\s+")]
enum LWsSplit {
    #[regex(r"\S+")]
    Run,
}
#[derive(Logos)]
#[logos(skip r"\s+")]
enum LWhitespace {
    #[regex(r"\w+")]
    Word,
    #[regex(r"[^\w\s]+")]
    Sym,
}
#[derive(Logos)]
enum LDigits {
    #[regex(r"\p{N}+")]
    Num,
    #[regex(r"[^\p{N}]+")]
    Other,
}

fn lex_count<'s, T: Logos<'s, Source = str>>(s: &'s str) -> usize
where
    T::Extras: Default,
{
    let mut lex = T::lexer(s);
    let mut n = 0;
    while lex.next().is_some() {
        n += 1;
    }
    n
}

/// logos throughput (ns/byte) for the engines it can express; `None` for the ones it can't (deepseek's
/// multi-split, and punct/bert whose POSIX-class isolation logos doesn't parse reliably).
fn logos_ns(ename: &str, s: &str, len: usize, iters: u32) -> Option<f64> {
    let f: fn(&str) -> usize = match ename {
        "gpt2" => |s| lex_count::<LGpt2>(s),
        "cl100k" => |s| lex_count::<LCl100k>(s),
        "o200k" => |s| lex_count::<LO200k>(s),
        "ws_split" => |s| lex_count::<LWsSplit>(s),
        "whitespace" => |s| lex_count::<LWhitespace>(s),
        "digits" => |s| lex_count::<LDigits>(s),
        _ => return None,
    };
    Some(ns_per_byte(len, iters, || f(s)))
}

// ── class-family pre-tokenizers: `class_runs_into` recipes + their equivalent reference regex ──
// Each is a monomorphized `Fsm` wrapper (const-generic masks can't go through a fn pointer directly).
fn f_ws_split(t: &[u8], g: &[u8], o: &mut [Span]) -> usize {
    class_runs_into::<{ mask::WS }, 0, 0>(t, g, o)
}
fn f_punct(t: &[u8], g: &[u8], o: &mut [Span]) -> usize {
    class_runs_into::<0, { mask::PUNCT }, 0>(t, g, o)
}
fn f_digits(t: &[u8], g: &[u8], o: &mut [Span]) -> usize {
    class_runs_into::<0, 0, { mask::NUMERIC }>(t, g, o)
}
fn f_whitespace(t: &[u8], g: &[u8], o: &mut [Span]) -> usize {
    class_runs_into::<{ mask::WS }, 0, { mask::WORD }>(t, g, o)
}
fn f_bert(t: &[u8], g: &[u8], o: &mut [Span]) -> usize {
    class_runs_into::<{ mask::WS }, { mask::PUNCT }, 0>(t, g, o)
}

// Class-family reference regexes — the exact split each recipe produces under a real regex engine.
// The char-classes map 1:1 to the atomsplit masks: WS=`\s`, WORD=`\w`, NUMERIC=`\p{N}`,
// PUNCT=`[[:punct:]\p{P}]` (ASCII-punctuation ∪ Unicode-P). Isolate = the punct alternative matches a
// single char; the run alternatives use `+`.
const RX_WS_SPLIT: &str = r"\S+";
const RX_WHITESPACE: &str = r"\w+|[^\w\s]+";
const RX_DIGITS: &str = r"\p{N}+|[^\p{N}]+";
const RX_PUNCT: &str = r"[[:punct:]\p{P}]|[^[:punct:]\p{P}]+";
const RX_BERT: &str = r"[[:punct:]\p{P}]|[^\s[:punct:]\p{P}]+";

// (name, native fsm, reference regex chain, keep_gaps, exact). The chain is applied `Isolated`, each
// regex splitting the previous pieces. `keep_gaps=false` for the whitespace-dropping recipes: their
// output is matches only (dropped whitespace is NOT a token), so the reference drops the gaps too.
// `exact`: the GPT FSMs are byte-exact with their regex (parity-tested) → ✓/✗ is a real gate. The
// class-family masks (WORD incl. Other_Alphabetic, PUNCT = P ∪ ASCII-symbols, …) can't be written as a
// regex class, so their reference is an *approximation* — a mismatch is `≈` (still a valid speed pairing),
// not a failure.
const ENGINES: &[(&str, Fsm, &[&str], bool, bool)] = &[
    ("gpt2", fsm_byte_level as Fsm, &[regexes::GPT2], true, true),
    ("cl100k", fsm_cl100k as Fsm, &[regexes::CL100K], true, true),
    ("o200k", fsm_o200k as Fsm, &[regexes::O200K], true, true),
    (
        "deepseek",
        fsm_deepseek as Fsm,
        regexes::DEEPSEEK,
        true,
        true,
    ),
    ("ws_split", f_ws_split as Fsm, &[RX_WS_SPLIT], false, false),
    (
        "whitespace",
        f_whitespace as Fsm,
        &[RX_WHITESPACE],
        false,
        false,
    ),
    ("digits", f_digits as Fsm, &[RX_DIGITS], true, false),
    ("punct", f_punct as Fsm, &[RX_PUNCT], true, false),
    ("bert", f_bert as Fsm, &[RX_BERT], false, false),
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

/// The composed split under `res` — each regex splits the previous pieces. `keep_gaps` keeps the
/// between/after-match gaps as pieces (Isolated split; deepseek); `false` drops them (the whitespace a
/// drop-ws recipe discards). Byte-offset pieces into `text`.
fn onig_pieces(res: &[Regex], text: &str, keep_gaps: bool) -> Vec<(usize, usize)> {
    let mut pieces = vec![(0usize, text.len())];
    for re in res {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for (s, e) in pieces.drain(..) {
            let sub = &text[s..e];
            let mut prev = 0usize;
            for (ms, me) in re.find_iter(sub) {
                if keep_gaps && ms > prev {
                    next.push((s + prev, s + ms));
                }
                next.push((s + ms, s + me));
                prev = me;
            }
            if keep_gaps && prev < sub.len() {
                next.push((s + prev, e));
            }
        }
        pieces = next;
    }
    pieces
}

/// Same composition under PCRE2 (JIT-compiled), operating on bytes; `find_iter` yields `Result<Match>`.
fn pcre2_pieces(res: &[Pcre2], text: &str, keep_gaps: bool) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut pieces = vec![(0usize, text.len())];
    for re in res {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for (s, e) in pieces.drain(..) {
            let sub = &bytes[s..e];
            let mut prev = 0usize;
            for m in re.find_iter(sub) {
                let Ok(m) = m else { break };
                let (ms, me) = (m.start(), m.end());
                if keep_gaps && ms > prev {
                    next.push((s + prev, s + ms));
                }
                next.push((s + ms, s + me));
                prev = me;
            }
            if keep_gaps && prev < sub.len() {
                next.push((s + prev, e));
            }
        }
        pieces = next;
    }
    pieces
}

/// Same composition under fancy-regex; `find_iter` yields `Result<Match>` (a match error ends the pass).
fn fancy_pieces(res: &[Fancy], text: &str, keep_gaps: bool) -> Vec<(usize, usize)> {
    let mut pieces = vec![(0usize, text.len())];
    for re in res {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for (s, e) in pieces.drain(..) {
            let sub = &text[s..e];
            let mut prev = 0usize;
            for m in re.find_iter(sub) {
                let Ok(m) = m else { break };
                let (ms, me) = (m.start(), m.end());
                if keep_gaps && ms > prev {
                    next.push((s + prev, s + ms));
                }
                next.push((s + ms, s + me));
                prev = me;
            }
            if keep_gaps && prev < sub.len() {
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
        "{:<9} {:<10} {:>7} {:>5}  {:>8} {:>8} | {:>8} | {:>8} {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7} {:>7}",
        "engine",
        "lang",
        "bytes",
        "b/tok",
        "clsSIMD",
        "clsScal",
        "fsm",
        "onig",
        "fancy",
        "logos",
        "pcre2jit",
        "vsOnig",
        "vsFncy",
        "vsLogos",
        "vsPcre2"
    );
    for &(ename, fsm, rxs, keep_gaps, exact) in ENGINES {
        let onig: Vec<Regex> = rxs.iter().map(|r| Regex::new(r).expect(ename)).collect();
        let fancy: Vec<Fancy> = rxs.iter().map(|r| Fancy::new(r).expect(ename)).collect();
        // PCRE2 with JIT — its speed is the JIT, so bench it there.
        let pcre2: Vec<Pcre2> = rxs
            .iter()
            .map(|r| {
                pcre2::bytes::RegexBuilder::new()
                    .utf(true)
                    .ucp(true)
                    .jit_if_available(true)
                    .build(r)
                    .expect(ename)
            })
            .collect();
        for (label, rel) in CORPORA {
            let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
                Ok(s) if !s.trim().is_empty() => s,
                _ => {
                    println!(
                        "{ename:<9} {label:<10}  (skipped — {rel} missing; run benches/data/fetch.py)"
                    );
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

            let ref_spans: Vec<Span> = onig_pieces(&onig, corpus, keep_gaps)
                .iter()
                .map(|&(s, e)| Span::new(s as u32, e as u32))
                .collect();
            let mut tags = vec![0u8; n];
            let mut tsc = vec![0u8; n];
            classify(text, &mut tags);
            let mut buf = vec![Span::default(); n + 1];
            let k = fsm(text, &tags, &mut buf);
            let ok = if buf[..k] == ref_spans[..] {
                "✓"
            } else if exact {
                "✗" // a real regression: the GPT fsm must be byte-exact with its regex
            } else {
                "≈" // class-family: regex is an approximation of the mask, divergence expected
            };
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
            let onig_ns = ns_per_byte(n, iters, || onig_pieces(&onig, corpus, keep_gaps).len());
            let fancy_ns = ns_per_byte(n, iters, || fancy_pieces(&fancy, corpus, keep_gaps).len());
            let pcre2_ns = ns_per_byte(n, iters, || pcre2_pieces(&pcre2, corpus, keep_gaps).len());
            let logos = logos_ns(ename, corpus, n, iters);

            let pipe = cls_simd + fsm_ns; // SIMD classify + scalar fsm — the full pipeline
            let (logos_c, vslogos) = match logos {
                Some(l) => (format!("{l:8.2}"), format!("{:6.1}x", l / pipe)),
                None => ("       —".into(), "      —".into()), // deepseek: no single logos grammar
            };
            println!(
                "{ename:<9} {label:<10} {n:>7} {btok:>5.1}  {cls_simd:>8.3} {cls_scal:>8.3} | {fsm_ns:>8.3} | {onig_ns:>8.2} {fancy_ns:>8.2} {logos_c} {pcre2_ns:>8.2} | {:>6.1}x {:>6.1}x {vslogos} {:>6.1}x {ok}",
                onig_ns / pipe,
                fancy_ns / pipe,
                pcre2_ns / pipe
            );
        }
    }
    println!(
        "\n(ns/byte, lower better. pipeline = SIMD classify + scalar fsm; vs onig / fancy / logos / \
         pcre2(JIT). GPT FSMs (gpt2/cl100k/o200k/deepseek): the regex IS the spec, ✓ = byte-exact, \
         ✗ = regression. Class family (ws_split/whitespace/digits/punct/bert): the mask can't be written \
         as a regex class, so the reference is approximate — ✓ where it happens to match, ≈ where it \
         diverges (speed still comparable). logos approximates the grammar (deepseek/punct/bert: n/a).)"
    );
}
