//! Reference-parity gates for the regex-shaped pre-tokenizers. Each of `fsm_cl100k` / `fsm_o200k` /
//! `fsm_byte_level` / `fsm_deepseek` must be BYTE-EXACT with the real reference — the oniguruma regex, composed exactly
//! as HF applies it (deepseek is a `Sequence` of three Isolated splits) — on a representative
//! multilingual corpus (ASCII, contractions, digits, punctuation runs, whitespace variants, Latin
//! accents/marks, Cyrillic/Greek/Arabic/Devanagari, Han/Kana/Hangul, ZWJ, astral emoji).
//!
//! This is the byte-exactness gate the hand cases in `fsm.rs` don't provide; the same corpus + gate
//! should run under x86 (Intel SDE) in CI to validate the SIMD paths.
//!
//! Gated off wasm32: the oniguruma reference is a C library that has no wasi libc to build against.
#![cfg(not(target_arch = "wasm32"))]
use atomsplit::classify::classify;
use atomsplit::fsm::{Span, fsm_byte_level, fsm_cl100k, fsm_deepseek, fsm_o200k};
use onig::Regex;

// tiktoken cl100k_base pre-tokenizer regex.
const CL100K: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]|\s+(?!\S)|\s+",
);
// GPT-2 / ByteLevel regex.
const GPT2: &str =
    r##"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"##;
// o200k_base / GPT-4o pre-tokenizer regex (case-aware letter runs + contraction suffix + `[\r\n/]` tail).
const O200K: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
);
// deepseek-v3 Sequence: `\p{N}{1,3}` → CJK-range → big regex, each Isolated.
const DS_NUM: &str = r"\p{N}{1,3}";
const DS_CJK: &str = r"[一-龥぀-ゟ゠-ヿ]+";
const DS_BIG: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

const CORPUS: &str = "The quick brown fox. Don't 12345 numbers, \u{00BD}\u{00B2}\u{00BC} \u{2168}! \
     café × naïve — Привет, наука! Ελλάδα 中文分词。ひらがな カタカナ 한글 مرحبا العربية \
     नरेंद्र मोदी x_y a1b2c3 e-mail@host.com 😀👍 hello  world\ttabs\nnewlines   end ";

fn spans(f: impl Fn(&[u8], &[u8], &mut [Span]) -> usize, s: &str) -> Vec<Span> {
    let mut tags = vec![0u8; s.len()];
    classify(s.as_bytes(), &mut tags);
    let mut out = vec![(0u32, 0u32); s.len() + 1];
    let k = f(s.as_bytes(), &tags, &mut out);
    out.truncate(k);
    out
}

fn onig_spans(re: &Regex, s: &str) -> Vec<Span> {
    re.find_iter(s).map(|(a, b)| (a as u32, b as u32)).collect()
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
    p3.into_iter().map(|(s, e)| (s as u32, e as u32)).collect()
}

#[test]
fn cl100k_parity() {
    let re = Regex::new(CL100K).unwrap();
    assert_eq!(spans(fsm_cl100k, CORPUS), onig_spans(&re, CORPUS));
}

#[test]
fn o200k_parity() {
    let re = Regex::new(O200K).unwrap();
    assert_eq!(spans(fsm_o200k, CORPUS), onig_spans(&re, CORPUS));
}

#[test]
fn byte_level_parity() {
    let re = Regex::new(GPT2).unwrap();
    assert_eq!(spans(fsm_byte_level, CORPUS), onig_spans(&re, CORPUS));
}

#[test]
fn deepseek_parity() {
    assert_eq!(spans(fsm_deepseek, CORPUS), deepseek_ref(CORPUS));
}
