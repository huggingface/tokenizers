//! Integration tests for the FSM pre-tokenizers. Kept out of `src/` so the core stays production-only.
use atomsplit::classify::{classify, mask};
use atomsplit::fsm::{
    CharDelimiterSplit, Span, class_runs_into, emit_class_spans, fsm_byte_level, fsm_cl100k,
    fsm_deepseek,
};

/// Run a no-push fsm into a fresh buffer and return the emitted spans.
fn spans(f: impl Fn(&[u8], &[u8], &mut [Span]) -> usize, s: &str) -> Vec<Span> {
    let mut tags = vec![0u8; s.len()];
    classify(s.as_bytes(), &mut tags);
    let mut out = vec![Span::default(); s.len() + 1];
    let k = f(s.as_bytes(), &tags, &mut out);
    out.truncate(k);
    out
}

#[test]
fn cl100k_rules() {
    // hand-verified against the tiktoken cl100k_base regex
    let cl = |s| spans(fsm_cl100k, s);
    assert_eq!(cl("Hello world"), vec![(0, 5), (5, 11)]); // "Hello" | " world"
    assert_eq!(cl("don't"), vec![(0, 3), (3, 5)]); // "don" | "'t" (contraction)
    assert_eq!(cl("a1234"), vec![(0, 1), (1, 4), (4, 5)]); // "a" | "123" | "4" ({1,3} cap)
    assert_eq!(cl("  hi"), vec![(0, 1), (1, 4)]); // " " | " hi"
    assert_eq!(cl("a, b"), vec![(0, 1), (1, 2), (2, 4)]); // "a" | "," | " b"
}

#[test]
fn deepseek_rules() {
    let ds = |s| spans(fsm_deepseek, s);
    assert_eq!(ds("abc中def"), vec![(0, 3), (3, 6), (6, 9)]); // letters | CJK | letters
    assert_eq!(ds("abc123"), vec![(0, 3), (3, 6)]); // letters | digits {1,3}
    assert_eq!(ds("_abc"), vec![(0, 4)]); // alt-1: ASCII punct + letters
    assert_eq!(ds("hello world"), vec![(0, 5), (5, 11)]); // word | space+word
    assert_eq!(ds("!!!"), vec![(0, 3)]); // \p{P}∪\p{S} run
}

#[test]
fn byte_level_rules() {
    let bl = |s| spans(fsm_byte_level, s);
    // ` ?\p{L}+`, lowercase contraction, ` ?\p{N}+` UNBOUNDED
    assert_eq!(bl("I'm 12345 ok"), vec![(0, 1), (1, 3), (3, 9), (9, 12)]);
    assert_eq!(bl("IT'S"), vec![(0, 2), (2, 3), (3, 4)]); // 'S is not a contraction (case-sensitive)
    assert_eq!(bl("hi   ok"), vec![(0, 2), (2, 4), (4, 7)]); // \s+(?!\S) leaves one space
}

#[test]
fn char_delimiter_split() {
    let mut out = vec![Span::default(); 8];
    // split on '/', Removed → drop delimiters, drop the empty gap between "//"
    let k = CharDelimiterSplit('/').pre_tokenize(b"a/bc//d", &mut [], &mut out);
    assert_eq!(&out[..k], &[(0, 1), (2, 4), (6, 7)]);
}

/// Byte-exactness gate for the class family: the NEON boundary extractor (`class_runs_into`) must equal
/// the scalar run-end core (`emit_class_spans`) for every recipe, at every char-aligned truncation length so
/// the < 16-byte NEON tail starts at every offset — including mid-char (chunk loop steps by 16). Corpus
/// mixes ASCII, 2/3-byte scripts, Devanagari letter+matra clusters, consecutive punct, tabs, astral.
#[test]
fn class_runs_into_matches() {
    let corpus = "Hello, world!! 123 café × наука 中文。। नरेंद्र मोदी ने ½²¼ ①② 😀a b\t".repeat(30);
    let full = corpus.as_bytes();
    let mut tags = vec![0u8; full.len()];
    let mut b1 = vec![Span::default(); full.len()];
    let mut b2 = vec![Span::default(); full.len()];

    fn eq<const D: u16, const I: u16, const A: u16>(
        t: &[u8],
        tg: &[u8],
        x: &mut [Span],
        y: &mut [Span],
        name: &str,
    ) {
        let k1 = class_runs_into::<D, I, A>(t, tg, x);
        let k2 = emit_class_spans::<D, I, A>(t, tg, y, 0, 0, 0, None);
        assert_eq!(&x[..k1], &y[..k2], "{} @len {}", name, t.len());
    }
    let mut sweep = |len: usize| {
        let (t, tg) = (&full[..len], &mut tags[..len]);
        classify(t, tg);
        let (x, y) = (&mut b1[..len], &mut b2[..len]);
        eq::<{ mask::WS }, 0, 0>(t, tg, x, y, "WhitespaceSplit");
        eq::<0, { mask::PUNCT }, 0>(t, tg, x, y, "Punctuation");
        eq::<0, 0, { mask::NUMERIC }>(t, tg, x, y, "Digits");
        eq::<{ mask::WS }, 0, { mask::WORD }>(t, tg, x, y, "Whitespace");
        eq::<{ mask::WS }, { mask::PUNCT }, 0>(t, tg, x, y, "Bert");
    };
    sweep(full.len());
    for c in full.len().saturating_sub(64)..full.len() {
        if corpus.is_char_boundary(c) {
            sweep(c);
        }
    }
}
