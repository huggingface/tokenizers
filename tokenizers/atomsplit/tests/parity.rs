//! Reference-parity gates for the regex-shaped pre-tokenizers. Each of `fsm_cl100k` / `fsm_o200k` /
//! `fsm_tekken` / `fsm_byte_level` / `fsm_deepseek` must be BYTE-EXACT with the real reference — the
//! oniguruma regex, composed exactly as HF applies it (deepseek is a `Sequence` of three Isolated
//! splits) — on two corpora: a multilingual one (ASCII, contractions, digits, punctuation runs,
//! whitespace variants, Latin accents/marks, Cyrillic/Greek/Arabic/Devanagari, Han/Kana/Hangul, ZWJ,
//! astral emoji) and an edge-case one (see [`EDGE`]).
//!
//! This is the byte-exactness gate the hand cases in `fsm.rs` don't provide; the same corpora + gate
//! should run under x86 (Intel SDE) in CI to validate the SIMD paths.
//!
//! Gated off wasm32: the oniguruma reference is a C library that has no wasi libc to build against.
#![cfg(not(target_arch = "wasm32"))]
use atomsplit::classify::classify;
use atomsplit::fsm::{Span, fsm_byte_level, fsm_cl100k, fsm_deepseek, fsm_o200k, fsm_tekken};
use onig::Regex;
// The oracle regexes are the canonical specs the FSMs implement — single source of truth in atomsplit.
use atomsplit::regexes::{
    CL100K, DEEPSEEK_BIG as DS_BIG, DEEPSEEK_CJK as DS_CJK, DEEPSEEK_NUM as DS_NUM, GPT2, O200K,
    TEKKEN,
};

const CORPUS: &str = "The quick brown fox. Don't 12345 numbers, \u{00BD}\u{00B2}\u{00BC} \u{2168}! \
     café × naïve — Привет, наука! Ελλάδα 中文分词。ひらがな カタカナ 한글 مرحبا العربية \
     नरेंद्र मोदी x_y a1b2c3 e-mail@host.com 😀👍 hello  world\ttabs\nnewlines   end ";

/// Second corpus, aimed at the axes where the o200k-shaped FSMs differ from each other: apostrophes
/// (contraction suffix vs plain prefix+letters), digit-run length (`{1,3}` vs one-per-token), and the
/// `[\r\n/]*` tail after a symbol run.
const EDGE: &str = "IT'S O'Brien can't 'quoted' l'été rock'n'roll\r\n\
     0 42 999 1000 1234567 v1.2.3 3.14159 1,000,000 \
     https://host/a/b?c=1&d=2 path/to//file /\r\n/ ///x \
     CamelCase XMLHttpRequest ĲSSELMEER ǅAMBO ŀl a\u{0301}b \
     日本語1234テスト ½3¼ \u{2168}42\u{2169} #tag @user $9.99 100% \
     end\n\n\nlines\r\n\r\n  \t  trailing   ";

fn spans(f: impl Fn(&[u8], &[u8], &mut [Span]) -> usize, s: &str) -> Vec<Span> {
    let mut tags = vec![0u8; s.len()];
    classify(s.as_bytes(), &mut tags);
    let mut out = vec![Span::default(); s.len() + 1];
    let k = f(s.as_bytes(), &tags, &mut out);
    out.truncate(k);
    out
}

fn onig_spans(re: &Regex, s: &str) -> Vec<Span> {
    re.find_iter(s)
        .map(|(a, b)| Span::new(a as u32, b as u32))
        .collect()
}

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

fn deepseek_ref(text: &str) -> Vec<Span> {
    let (rn, rc, rb) = (
        Regex::new(DS_NUM).unwrap(),
        Regex::new(DS_CJK).unwrap(),
        Regex::new(DS_BIG).unwrap(),
    );
    let mut p1 = Vec::new();
    split_iso(text, 0, text.len(), &rn, &mut p1);
    let mut p2 = Vec::new();
    for (s, e) in p1 {
        split_iso(text, s, e, &rc, &mut p2);
    }
    let mut p3 = Vec::new();
    for (s, e) in p2 {
        split_iso(text, s, e, &rb, &mut p3);
    }
    p3.into_iter()
        .map(|(s, e)| Span::new(s as u32, e as u32))
        .collect()
}

/// Both corpora, one regex: the FSM must reproduce the oracle span-for-span.
fn check(fsm: impl Fn(&[u8], &[u8], &mut [Span]) -> usize + Copy, pattern: &str) {
    let re = Regex::new(pattern).unwrap();
    for text in [CORPUS, EDGE] {
        assert_eq!(spans(fsm, text), onig_spans(&re, text), "{text:?}");
    }
}

#[test]
fn cl100k_parity() {
    check(fsm_cl100k, CL100K);
}

#[test]
fn o200k_parity() {
    check(fsm_o200k, O200K);
}

#[test]
fn tekken_parity() {
    check(fsm_tekken, TEKKEN);
}

#[test]
fn byte_level_parity() {
    check(fsm_byte_level, GPT2);
}

#[test]
fn deepseek_parity() {
    for text in [CORPUS, EDGE] {
        assert_eq!(spans(fsm_deepseek, text), deepseek_ref(text), "{text:?}");
    }
}
