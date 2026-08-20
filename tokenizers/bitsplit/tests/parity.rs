//! Byte-exactness gate for the bitstream splitters. Oracle = oniguruma, composed exactly as HF
//! applies it (deepseek is a `Sequence` of three Isolated splits, not one regex).
//!
//! - The SWEEP is the point. A bitstream program is only interesting where a rule straddles the
//!   64-byte grid: `starts[bi-1] |= patch`, the `anl` retraction, every `pb`/`nb` edge peek. Slicing
//!   the corpus at every char boundary walks every construct through every block phase, and the
//!   oracle re-runs on the same slice so the expectation is never hand-written.
//! - `find_iter` is the Isolated split only because these regexes match the whole input with no
//!   gaps; deepseek's three passes do leave gaps, hence `split_iso`.
#![cfg(not(target_arch = "wasm32"))]

use bitsplit::Span;
use bitsplit::classify::classify;
use bitsplit::regexes::{
    CL100K, DEEPSEEK_BIG as DS_BIG, DEEPSEEK_CJK as DS_CJK, DEEPSEEK_NUM as DS_NUM, GPT2, KIMI_K2,
    O200K, TEKKEN,
};
use onig::Regex;

const CORPUS: &str = "The quick brown fox. Don't 12345 numbers, \u{00BD}\u{00B2}\u{00BC} \u{2168}! \
     café × naïve — Привет, наука! Ελλάδα 中文分词。ひらがな カタカナ 한글 مرحبا العربية \
     नरेंद्र मोदी x_y a1b2c3 e-mail@host.com 😀👍 hello  world\ttabs\nnewlines   end ";

const EDGE: &str = "IT'S O'Brien can't 'quoted' l'été rock'n'roll\r\n\
     0 42 999 1000 1234567 v1.2.3 3.14159 1,000,000 \
     https://host/a/b?c=1&d=2 path/to//file /\r\n/ ///x \
     CamelCase XMLHttpRequest ĲSSELMEER ǅAMBO ŀl a\u{0301}b \
     日本語1234テスト ½3¼ \u{2168}42\u{2169} #tag @user $9.99 100% \
     end\n\n\nlines\r\n\r\n  \t  trailing   ";

/// A bitstream splitter, normalised to one shape (`starts` + `flag` + `later` scratch; only the
/// o200k family reads `later`, and deepseek ignores `flag` too).
type Split = fn(&[u8], &[u8], &mut [u64], &mut [u64], &mut [u64], &mut [Span]) -> usize;

fn bs_deepseek(t: &[u8], g: &[u8], s: &mut [u64], _f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_deepseek(t, g, s, o)
}
fn bs_byte_level(t: &[u8], g: &[u8], s: &mut [u64], f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_byte_level(t, g, s, f, o)
}
fn bs_cl100k(t: &[u8], g: &[u8], s: &mut [u64], f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_cl100k(t, g, s, f, o)
}
fn bs_qwen(t: &[u8], g: &[u8], s: &mut [u64], f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_qwen(t, g, s, f, o)
}
fn bs_o200k(t: &[u8], g: &[u8], s: &mut [u64], f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_o200k(t, g, s, f, _l, o)
}
fn bs_tekken(t: &[u8], g: &[u8], s: &mut [u64], f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_tekken(t, g, s, f, _l, o)
}
fn bs_kimi(t: &[u8], g: &[u8], s: &mut [u64], f: &mut [u64], _l: &mut [u64], o: &mut [Span]) -> usize {
    bitsplit::bitsplit_kimi(t, g, s, f, _l, o)
}

fn spans(f: Split, s: &str) -> Vec<Span> {
    if s.is_empty() {
        return Vec::new();
    }
    let b = s.as_bytes();
    let mut tags = vec![0u8; b.len()];
    classify(b, &mut tags);
    let nblk = b.len().div_ceil(64);
    let (mut starts, mut flag) = (vec![0u64; nblk], vec![0u64; nblk]);
    let mut later = vec![0u64; 2 * nblk];
    let mut out = vec![Span::default(); b.len() + 1];
    let k = f(b, &tags, &mut starts, &mut flag, &mut later, &mut out);
    out.truncate(k);
    out
}

fn onig_spans(re: &Regex, s: &str) -> Vec<Span> {
    re.find_iter(s)
        .map(|(a, b)| Span::new(a as u32, b as u32))
        .collect()
}

/// One Isolated split of `text[s..e]`: gaps + matches, absolute offsets.
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

fn deepseek_ref(text: &str) -> Vec<Span> {
    let (rn, rc, rb) = (
        Regex::new(DS_NUM).unwrap(),
        Regex::new(DS_CJK).unwrap(),
        Regex::new(DS_BIG).unwrap(),
    );
    let mut a = Vec::new();
    split_iso(text, 0, text.len(), &rn, &mut a);
    let mut b = Vec::new();
    for (s, e) in a {
        split_iso(text, s, e, &rc, &mut b);
    }
    let mut c = Vec::new();
    for (s, e) in b {
        split_iso(text, s, e, &rb, &mut c);
    }
    c.into_iter()
        .map(|(s, e)| Span::new(s as u32, e as u32))
        .collect()
}

/// The oracle for a grammar, over an arbitrary slice.
enum Oracle {
    Whole(Regex),
    DeepSeek,
}

impl Oracle {
    fn spans(&self, s: &str) -> Vec<Span> {
        match self {
            Oracle::Whole(re) => onig_spans(re, s),
            Oracle::DeepSeek => deepseek_ref(s),
        }
    }
}

fn long_corpus() -> String {
    let mut s = String::new();
    while s.len() < 4096 {
        s.push_str(CORPUS);
        s.push_str(EDGE);
    }
    s
}

/// Both corpora whole, then the block-phase sweep: every char-boundary prefix and suffix of a
/// >4 KB text, so each rule crosses a block edge in every alignment.
fn check(name: &str, f: Split, oracle: &Oracle) {
    for text in [CORPUS, EDGE] {
        assert_eq!(spans(f, text), oracle.spans(text), "{name}: whole {text:?}");
    }

    let long = long_corpus();
    let mut checked = 0usize;

    // suffixes: shifts the whole text across the grid
    for off in 0..512.min(long.len()) {
        if !long.is_char_boundary(off) {
            continue;
        }
        let sub = &long[off..];
        assert_eq!(spans(f, sub), oracle.spans(sub), "{name}: suffix off={off}");
        checked += 1;
    }
    // prefixes: exercises the last-block / EOF rules (`\s+(?!\S)` vs plain `\s+`) in every phase
    for end in (long.len().saturating_sub(512))..=long.len() {
        if !long.is_char_boundary(end) {
            continue;
        }
        let sub = &long[..end];
        assert_eq!(spans(f, sub), oracle.spans(sub), "{name}: prefix end={end}");
        checked += 1;
    }
    assert!(checked > 500, "{name}: sweep too small ({checked})");
}

/// Deterministic pseudo-random text from a weighted alphabet — always valid UTF-8 by construction.
/// Aimed at the carry logic: long digit runs, whitespace/newline runs and apostrophes next to
/// letters are what the `\p{N}{1,3}` grouping, the `\s*[\r\n]+` fill and the contraction escape
/// disagree about.
fn fuzz_texts(n: usize) -> Vec<String> {
    const ALPHA: &[&str] = &[
        "a", "b", "z", "A", "Q", "é", "ß", "Ĳ", "0", "1", "9", " ", "  ", "\n", "\r\n", "\t", "'",
        "'s", "'ll", ".", "!", "/", "#", "_", "中", "文", "ひ", "カ", "한", "م", "😀", "\u{0301}",
        "\u{200D}", "½", "\u{2168}",
    ];
    let mut st = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    (0..n)
        .map(|_| {
            let len = 1 + (next() % 300) as usize;
            (0..len)
                .map(|_| ALPHA[(next() % ALPHA.len() as u64) as usize])
                .collect()
        })
        .collect()
}

fn check_fuzz(name: &str, f: Split, oracle: &Oracle) {
    for (i, t) in fuzz_texts(4000).iter().enumerate() {
        assert_eq!(spans(f, t), oracle.spans(t), "{name}: fuzz #{i} {t:?}");
    }
}

#[test]
fn byte_level_parity() {
    let o = Oracle::Whole(Regex::new(GPT2).unwrap());
    check("byte_level", bs_byte_level, &o);
    check_fuzz("byte_level", bs_byte_level, &o);
}

#[test]
fn cl100k_parity() {
    let o = Oracle::Whole(Regex::new(CL100K).unwrap());
    check("cl100k", bs_cl100k, &o);
    check_fuzz("cl100k", bs_cl100k, &o);
}

/// Qwen2 / Qwen3: cl100k with a bare `\p{N}`.
#[test]
fn qwen_parity() {
    let qwen = CL100K.replace(r"\p{N}{1,3}", r"\p{N}");
    let o = Oracle::Whole(Regex::new(&qwen).unwrap());
    check("qwen", bs_qwen, &o);
    check_fuzz("qwen", bs_qwen, &o);
}

#[test]
fn deepseek_parity() {
    check("deepseek", bs_deepseek, &Oracle::DeepSeek);
    check_fuzz("deepseek", bs_deepseek, &Oracle::DeepSeek);
}

/// o200k_base / GPT-4o — and byte-for-byte the regex Llama-4, gpt-oss and MiniMax-M2 ship, so this
/// one test covers four families.
#[test]
fn o200k_parity() {
    let o = Oracle::Whole(Regex::new(O200K).unwrap());
    check("o200k", bs_o200k, &o);
    check_fuzz("o200k", bs_o200k, &o);
}

#[test]
fn tekken_parity() {
    let o = Oracle::Whole(Regex::new(TEKKEN).unwrap());
    check("tekken", bs_tekken, &o);
    check_fuzz("tekken", bs_tekken, &o);
}

#[test]
fn kimi_parity() {
    let o = Oracle::Whole(Regex::new(KIMI_K2).unwrap());
    check("kimi", bs_kimi, &o);
    check_fuzz("kimi", bs_kimi, &o);
}

/// Negative control: the gate above only means something if it can fail. gpt2 and cl100k disagree
/// on plenty (`\p{N}{1,3}`, the `[\r\n]*` tail, the non-space `?` prefix), so crossing them must
/// blow up — if this ever passes, `check` has stopped comparing anything.
#[test]
#[should_panic(expected = "negative-control")]
fn harness_discriminates() {
    check(
        "negative-control",
        bs_byte_level,
        &Oracle::Whole(Regex::new(CL100K).unwrap()),
    );
}
