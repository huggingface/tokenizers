//! Integration tests for the FSM pre-tokenizers. Kept out of `src/` so the core stays production-only.
use atomsplit::classify::{classify, mask};
use atomsplit::fsm::{
    Span, class_runs_into, emit_class_spans, fsm_byte_level, fsm_cl100k, fsm_deepseek, fsm_o200k,
    fsm_tekken, scan_byte_level, scan_byte_level_masked, scan_cl100k_cap, scan_cl100k_cap_masked,
    scan_deepseek, scan_deepseek_masked, scan_o200k, scan_o200k_masked, scan_tekken,
    scan_tekken_masked,
};

/// The shared sweep for a masked/scalar scanner pair: compare spans on the corpus behind
/// every 64-byte-edge offset (leading padding 0..=70), and with truncations exercising the
/// scalar tail at every remaining length.
type ScanFn = dyn Fn(&[u8], &[u8], &mut dyn FnMut(Span));

fn masked_matches_scalar(corpus: &str, scalar: &ScanFn, masked: &ScanFn) {
    fn spans_of(scan: &ScanFn, s: &[u8]) -> Vec<Span> {
        let mut tags = vec![0u8; s.len()];
        classify(s, &mut tags);
        let mut v = Vec::new();
        scan(s, &tags, &mut |sp| v.push(sp));
        v
    }
    let check = |s: &str| {
        let a = spans_of(scalar, s.as_bytes());
        let b = spans_of(masked, s.as_bytes());
        assert_eq!(b, a, "input len {}: {:?}", s.len(), s);
    };
    for pad in 0..=70 {
        check(&format!("{}{}", "x".repeat(pad), corpus));
    }
    for pad in [0, 37] {
        let padded = format!("{}{}", "x".repeat(pad), corpus);
        for len in padded.len().saturating_sub(140)..=padded.len() {
            if padded.is_char_boundary(len) {
                check(&padded[..len]);
            }
        }
    }
    for len in 0..=70 {
        if corpus.is_char_boundary(len) {
            check(&corpus[..len]);
        }
    }
}

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

/// Mistral's tekken split is o200k's, minus the contraction suffix, with one token per digit.
#[test]
fn tekken_rules() {
    let tk = |s| spans(fsm_tekken, s);
    assert_eq!(tk("don't"), vec![(0, 3), (3, 5)]); // "don" | "'t" — prefix+letters, not a contraction
    assert_eq!(spans(fsm_o200k, "don't"), vec![(0, 5)]); // o200k glues the contraction on
    assert_eq!(tk("a1234"), vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]); // `\p{N}`, one digit each
    assert_eq!(tk("XMLHttpRequest"), vec![(0, 7), (7, 14)]); // case split: "XMLHttp" | "Request"
    assert_eq!(tk("a/\r\nb"), vec![(0, 1), (1, 4), (4, 5)]); // "/" run + its `[\r\n/]*` tail
    assert_eq!(tk("hi   ok"), vec![(0, 2), (2, 4), (4, 7)]); // \s+(?!\S) leaves one space
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

/// Byte-exactness gate for the masked byte-level scanner: `scan_byte_level_masked` must emit the
/// spans `scan_byte_level` emits, on every input. The corpus stresses every rule shape the batch
/// algebra rewrites: contractions in both cases and at rejection edges, prefix spaces before all
/// three run classes, unbounded numbers including non-ASCII digits, multi-byte whitespace (the
/// bad-zone route), CJK/emoji/ZWJ runs, CRLF and tab runs, and runs longer than a batch. The
/// padding sweep (0..=70 leading bytes) moves every shape across every 64-byte batch edge, so
/// batch-edge carries, the bit-63 lookahead and every bad-zone route are hit at every offset; the
/// truncation sweep exercises the scalar tail at every remaining length.
#[test]
fn masked_scan_matches_scan_byte_level() {
    let corpus = concat!(
        "I'm 12345 ok, don't they'll 've 'lx x's ''s IT'S 's mid'dle end' ",
        "hello   world\r\nmore\ttabs\t\t  spaced  end ",
        "no.1 中文漢字テスト مرحبا १२३४५६७८९० ¹²³ ½¾ ",
        "emoji 😀😀 zwj 👩\u{200d}🔬 nbsp\u{a0}x thin\u{2009}y wide\u{3000}z ",
        "((()))!!!??? #$%&' “curly” apostrophe’d ",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
        "9999999999999999999999999999999999999999999999999999999999999999999999999999999 ",
        "中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中                    end",
    );
    masked_matches_scalar(
        corpus,
        &|t, tg, e| scan_byte_level(t, tg, e),
        &|t, tg, e| scan_byte_level_masked(t, tg, e),
    );
}

/// Byte-exactness gate for the masked cl100k-family scanner, at every shipped digit cap. On top
/// of the byte-level shapes: capped digit runs (pure ASCII, pure Devanagari, and mixed, so the
/// char-counted bad route is hit mid-run), letters after punct at run starts vs mid-run (the
/// two-chars-back absorb test), newlines absorbed after punct runs (`[\r\n]*`), whitespace runs
/// with interior newlines (`\s*[\r\n]`), tab prefixes, and an apostrophe before a non-ASCII
/// char (the `'ſ` defer).
#[test]
fn masked_scan_matches_scan_cl100k_cap() {
    let corpus = concat!(
        "I'm 12345 ok, don't they'LL 've 'lx x's ''s IT'S 's 3'ts end' ",
        "a1234 12٣45 ١٢٣٤٥٦ १२३४५६७८९० 999999999999999999999999999999999999999999999999 ",
        "!!a x!a ?!x. ;\n\n};\r\n\r\n foo();\nbar() //x\n\n\n end\n",
        "hello   world\r\nmore\ttabs\t\t  spaced \ta \r\n \n\t\r\n\t x ",
        "no.1 中文漢字テスト مرحبا nbsp\u{a0}x thin\u{2009}y wide\u{3000}z 'ſok “curly”’d ",
        "((()))!!!??? #$%&' 😀😀 👩\u{200d}🔬 ",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
        "中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中中                    end",
    );
    for cap in [3, 1, usize::MAX] {
        masked_matches_scalar(
            corpus,
            &move |t, tg, e| scan_cl100k_cap(t, tg, cap, e),
            &move |t, tg, e| scan_cl100k_cap_masked(t, tg, cap, e),
        );
    }
}

/// Byte-exactness gate for the masked o200k/tekken scanner. On top of the cl100k shapes: case
/// splits (camelCase, all-upper runs, upper after caseless — the deferred backtrack), suffix
/// contractions incl. chains ("can'ts", "x'll'd") and prefix apostrophes after digits, `[\r\n/]*`
/// tails with slash runs before and after newlines (the walkback shapes), and combining marks
/// (run-contextual class, the wide bad smear).
#[test]
fn masked_scan_matches_scan_o200k_and_tekken() {
    let corpus = concat!(
        "camelCase HTTPResponse XMLHttpRequest AAAA aaaa aA Aa 中B B中b ʰupper Xʰa 中中中中 ",
        "don't they'LL CAN'TS x'll'd 3'ts I'm'll a'sit 's'' end' 'ſok ",
        "a1234 12٣45 १२३४५६ 99999999999999999999999999999999999999999999999999999999999999999 ",
        "foo();\nbar() .\n// ///x\r\n/// a/b//c\n\n//d e\u{301}f g\u{5bf}h zwj\u{200c}x ",
        "hello   world\r\nmore\ttabs\t\t  spaced \ta \r\n \n\t\r\n\t x nbsp\u{a0}x wide\u{3000}z ",
        "((()))!!!??? #$%&' “curly”’d 😀😀 مرحبا מבחן ",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa end"
    );
    masked_matches_scalar(
        corpus,
        &|t, tg, e| scan_o200k(t, tg, e),
        &|t, tg, e| scan_o200k_masked(t, tg, e),
    );
    masked_matches_scalar(
        corpus,
        &|t, tg, e| scan_tekken(t, tg, e),
        &|t, tg, e| scan_tekken_masked(t, tg, e),
    );
}

/// Byte-exactness gate for the masked deepseek scanner. On top of the shared shapes: CJK
/// letter/punct runs and their neighborhoods (the closed-unit rule and the bad cover), gap runs
/// (controls / NumericOther / ZWJ) with and without a following letter run (the prefix split),
/// alt-1 `[ascii-punct][A-Za-z]+` including collisions with non-ASCII letters ("_naïve"), and
/// whitespace runs followed by digits or CJK (no give-back).
#[test]
fn masked_scan_matches_scan_deepseek() {
    let corpus = concat!(
        "abc中def 中文漢字テスト!ひらがな・カタカナ 中中中中中中中 mixed中123中ok 拼音列表(e.g. 表!x ",
        "_abc (foo) .py x!a _naïve _né !!a a/b.c 3'ts don't ½x ¼¼y\u{7f}z zwj\u{200c}gap\u{200c}\u{200c}word ",
        "a1234 12٣45 999999999999999999999999999999999999999999999999999999999999999999 12 中 ",
        "x   123 y\t\t\t45 z  \n  99 w   中 Model):\n    money = f(x):\n\n\t  indent ",
        "hello   world\r\nmore\ttabs\t\t  spaced \ta \r\n \n\t\r\n\t x wide\u{3000}z 中 12\n\n34 ",
        "((()))!!!??? #$%&' “curly”’d 😀😀 مرحبا ",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa end"
    );
    masked_matches_scalar(
        corpus,
        &|t, tg, e| scan_deepseek(t, tg, e),
        &|t, tg, e| scan_deepseek_masked(t, tg, e),
    );
}

#[test]
fn byte_level_rules() {
    let bl = |s| spans(fsm_byte_level, s);
    // ` ?\p{L}+`, lowercase contraction, ` ?\p{N}+` UNBOUNDED
    assert_eq!(bl("I'm 12345 ok"), vec![(0, 1), (1, 3), (3, 9), (9, 12)]);
    assert_eq!(bl("IT'S"), vec![(0, 2), (2, 3), (3, 4)]); // 'S is not a contraction (case-sensitive)
    assert_eq!(bl("hi   ok"), vec![(0, 2), (2, 4), (4, 7)]); // \s+(?!\S) leaves one space
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
