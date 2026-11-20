//! Soundness gate for [`bitcannon::cuts`]: at every reported cut, splitting the text and
//! pre-tokenizing both halves must give the whole text's spans, per oniguruma and per the bitstream
//! grammar. Over the corpora, a block-phase sweep and random strings.
#![cfg(not(target_arch = "wasm32"))]

use bitcannon::Span;
use bitcannon::classify::classify;
use bitcannon::cuts::after_letter;
use bitcannon::regexes::{CL100K, GPT2};
use onig::Regex;

const CORPUS: &str = "The quick brown fox. Don't 12345 numbers, \u{00BD}\u{00B2}\u{00BC} \u{2168}! \
     café × naïve — Привет, наука! Ελλάδα 中文分词。ひらがな カタカナ 한글 مرحبا العربية \
     नरेंद्र मोदी x_y a1b2c3 e-mail@host.com 😀👍 hello  world\ttabs\nnewlines   end ";

const EDGE: &str = "IT'S O'Brien can't 'quoted' l'été rock'n'roll\r\n\
     0 42 999 1000 1234567 v1.2.3 3.14159 1,000,000 \
     https://host/a/b?c=1&d=2 path/to//file /\r\n/ ///x \
     CamelCase XMLHttpRequest ĲSSELMEER ǅAMBO ŀl a\u{0301}b \
     日本語1234テスト ½3¼ \u{2168}42\u{2169} #tag @user $9.99 100% \
     end\n\n\nlines\r\n\r\n  \t  trailing   x'll y'LL z'Re don'T";

type Grammar = fn(&[u8], &[u8], &mut [u64], &mut [u64], &mut [Span]) -> usize;

fn grammar_spans(f: Grammar, s: &str) -> Vec<Span> {
    if s.is_empty() {
        return Vec::new();
    }
    let b = s.as_bytes();
    let mut tags = vec![0u8; b.len()];
    classify(b, &mut tags);
    let nblk = b.len().div_ceil(64);
    let (mut starts, mut flag) = (vec![0u64; nblk], vec![0u64; nblk]);
    let mut out = vec![Span::default(); b.len() + 1];
    let k = f(b, &tags, &mut starts, &mut flag, &mut out);
    out.truncate(k);
    out
}

fn onig_spans(re: &Regex, s: &str) -> Vec<Span> {
    re.find_iter(s)
        .map(|(a, b)| Span::new(a as u32, b as u32))
        .collect()
}

/// Every cut `after_letter` reports in `text`.
fn cuts(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(p) = after_letter(text.as_bytes(), from, text.len()) {
        out.push(p);
        from = p + 1;
        while !text.is_char_boundary(from) {
            from += 1;
        }
    }
    out
}

/// `split(text) == split(text[..p]) ++ split(text[p..]) + p` at every cut.
fn check_cuts(name: &str, text: &str, split: &dyn Fn(&str) -> Vec<Span>) -> usize {
    let whole = split(text);
    let cuts = cuts(text);
    for &p in &cuts {
        let mut halves = split(&text[..p]);
        halves.extend(
            split(&text[p..])
                .into_iter()
                .map(|s| Span::new(s.start + p as u32, s.end + p as u32)),
        );
        assert_eq!(halves, whole, "{name}: cut at {p} in {text:?}");
    }
    cuts.len()
}

/// The grammars `after_letter` claims, each with its regex and its bitstream program.
fn grammars() -> Vec<(&'static str, Regex, Grammar)> {
    let qwen = CL100K.replace(r"\p{N}{1,3}", r"\p{N}");
    vec![
        (
            "gpt2",
            Regex::new(GPT2).unwrap(),
            bitcannon::bitcannon_byte_level,
        ),
        (
            "cl100k",
            Regex::new(CL100K).unwrap(),
            bitcannon::bitcannon_cl100k,
        ),
        (
            "qwen",
            Regex::new(&qwen).unwrap(),
            bitcannon::bitcannon_qwen,
        ),
    ]
}

#[test]
fn after_letter_is_sound_on_the_corpora() {
    for (name, re, grammar) in grammars() {
        for text in [CORPUS, EDGE] {
            let n = check_cuts(name, text, &|s| onig_spans(&re, s));
            assert!(n > 30, "{name}: only {n} cuts in {text:?}");
            check_cuts(name, text, &|s| grammar_spans(grammar, s));
        }
    }
}

/// Suffixes of a >4KB text, so cuts land in every 64-byte block phase (bitstream grammar only).
#[test]
fn after_letter_is_sound_across_block_phases() {
    let mut long = String::new();
    while long.len() < 4096 {
        long.push_str(CORPUS);
        long.push_str(EDGE);
    }
    for (name, _, grammar) in grammars() {
        let mut checked = 0;
        for off in (0..130).filter(|&o| long.is_char_boundary(o)) {
            checked += check_cuts(name, &long[off..], &|s| grammar_spans(grammar, s));
        }
        assert!(checked > 10_000, "{name}: only {checked} cuts checked");
    }
}

/// Short random strings over the chars the grammars branch on.
#[test]
fn after_letter_is_sound_on_random_strings() {
    const ALPHABET: &[&str] = &[
        "a", "Z", "s", "t", "r", "e", "v", "m", "l", "d", "S", "T", "L", "é", "ß", "Ж", "中", "ひ",
        "'", "'", "1", "7", "½", "Ⅸ", "٣", " ", " ", "  ", "\n", "\r\n", "\t", "\u{00A0}", "!",
        ".", ",", "-", "_", "/", "\u{0301}", "\u{200D}", "😀", "€", "\u{0007}",
    ];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let grammars = grammars();
    let mut checked = 0;
    for _ in 0..20_000 {
        let len = 1 + (next() % 24) as usize;
        let text: String = (0..len)
            .map(|_| ALPHABET[(next() % ALPHABET.len() as u64) as usize])
            .collect();
        for (name, re, grammar) in &grammars {
            checked += check_cuts(name, &text, &|s| onig_spans(re, s));
            check_cuts(name, &text, &|s| grammar_spans(*grammar, s));
        }
    }
    assert!(checked > 50_000, "only {checked} cuts checked");
}

/// Negative control: unsound for o200k, so the harness can fail.
#[test]
fn after_letter_is_unsound_for_o200k() {
    let re = Regex::new(bitcannon::regexes::O200K).unwrap();
    let text = "don't";
    let p = after_letter(text.as_bytes(), 0, text.len()).unwrap();
    let mut halves = onig_spans(&re, &text[..p]);
    halves.extend(
        onig_spans(&re, &text[p..])
            .into_iter()
            .map(|s| Span::new(s.start + p as u32, s.end + p as u32)),
    );
    assert_ne!(halves, onig_spans(&re, text));
}
