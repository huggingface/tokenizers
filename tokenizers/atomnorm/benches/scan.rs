//! Scan normalizers (lowercase / strip_accents / nmt / bert) vs the legacy tk-encode char-iterator
//! implementations, on the Wikipedia corpora. run: cargo bench -p atomnorm --bench scan
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

fn ref_lowercase(input: &str) -> Cow<'_, str> {
    let lowercases_to_self = |c: char| {
        let mut it = c.to_lowercase();
        matches!((it.next(), it.next()), (Some(first), None) if first == c)
    };
    if input.chars().all(lowercases_to_self) {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.chars().flat_map(|c| c.to_lowercase()).collect())
    }
}

fn ref_strip_accents(input: &str) -> Cow<'_, str> {
    use unicode_normalization_alignments::char::is_combining_mark;
    if input.chars().any(is_combining_mark) {
        Cow::Owned(input.chars().filter(|&c| !is_combining_mark(c)).collect())
    } else {
        Cow::Borrowed(input)
    }
}

fn nmt_removes(c: char) -> bool {
    matches!(c as u32,
        0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F)
}
fn nmt_to_space(c: char) -> char {
    match c as u32 {
        0x0009
        | 0x000A
        | 0x000C
        | 0x000D
        | 0x1680
        | 0x200B..=0x200F
        | 0x2028
        | 0x2029
        | 0x2581
        | 0xFEFF
        | 0xFFFD => ' ',
        _ => c,
    }
}
fn ref_nmt(input: &str) -> Cow<'_, str> {
    if input
        .chars()
        .any(|c| nmt_removes(c) || nmt_to_space(c) != c)
    {
        Cow::Owned(
            input
                .chars()
                .filter(|&c| !nmt_removes(c))
                .map(nmt_to_space)
                .collect(),
        )
    } else {
        Cow::Borrowed(input)
    }
}

// the legacy BertNormalizer pipeline path, verbatim (default flags: all on, strip = lowercase)
fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}
fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}
fn clean_text_removes(c: char) -> bool {
    c == '\0' || c == '\u{fffd}' || is_control(c)
}
fn is_chinese_char(c: char) -> bool {
    matches!(c as usize,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F |
        0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}
fn ref_bert(input: &str) -> Cow<'_, str> {
    use unicode_normalization::{IsNormalized, is_nfd_quick};
    let lowercases_to_self = |c: char| {
        let mut it = c.to_lowercase();
        matches!((it.next(), it.next()), (Some(first), None) if first == c)
    };
    let noop = matches!(is_nfd_quick(input.chars()), IsNormalized::Yes)
        && !input.chars().any(|c| {
            clean_text_removes(c)
                || (is_whitespace(c) && c != ' ')
                || is_chinese_char(c)
                || c.is_mark_nonspacing()
                || !lowercases_to_self(c)
        });
    if noop {
        return Cow::Borrowed(input);
    }
    let cleaned = input
        .chars()
        .filter(|&c| !clean_text_removes(c))
        .flat_map(|c| {
            let c = if is_whitespace(c) { ' ' } else { c };
            if is_chinese_char(c) {
                [Some(' '), Some(c), Some(' ')]
            } else {
                [Some(c), None, None]
            }
        })
        .flatten();
    let mut normalized = String::with_capacity(input.len());
    normalized.extend(
        cleaned
            .nfd()
            .filter(|c| !c.is_mark_nonspacing())
            .flat_map(char::to_lowercase),
    );
    Cow::Owned(normalized)
}

fn best(len: usize, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        b = b.min(t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64);
    }
    b
}

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "../atomsplit/benches/data/fr.txt"),
    ("Russian", "../atomsplit/benches/data/ru.txt"),
    ("Greek", "../atomsplit/benches/data/el.txt"),
    ("Hebrew", "../atomsplit/benches/data/he.txt"),
    ("Arabic", "../atomsplit/benches/data/ar.txt"),
    ("Hindi", "../atomsplit/benches/data/hi.txt"),
    ("Thai", "../atomsplit/benches/data/th.txt"),
    ("Chinese", "../atomsplit/benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "../atomsplit/benches/data/ko.txt"),
];

type F = fn(&str) -> Cow<'_, str>;
fn bert_simd(s: &str) -> Cow<'_, str> {
    atomnorm::bert(s, true, true, true, true)
}
fn bert_sclr(s: &str) -> Cow<'_, str> {
    atomnorm::scalar::bert(s, true, true, true, true)
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    println!(
        "\nns/byte, lower=better. simd/sclr = atomnorm scan paths; legacy = the tk-encode char-iterator\n\
         implementation. vsLeg = legacy / simd. exact = both paths byte-identical to legacy.\n"
    );
    println!(
        "{:<6} {:<9} {:>7} {:>7} {:>7} {:>8} | {:>7} {:>6}",
        "norm", "lang", "bytes", "simd", "sclr", "legacy", "vsLeg", "exact"
    );
    let norms: &[(&str, F, F, F)] = &[
        (
            "lower",
            atomnorm::lowercase as F,
            atomnorm::scalar::lowercase as F,
            ref_lowercase as F,
        ),
        (
            "strip",
            atomnorm::strip_accents,
            atomnorm::scalar::strip_accents,
            ref_strip_accents,
        ),
        ("nmt", atomnorm::nmt, atomnorm::scalar::nmt, ref_nmt),
        ("bert", bert_simd, bert_sclr, ref_bert),
    ];
    for (name, simd, sclr, legacy) in norms {
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
            let text = &s[..c];
            let n = text.len();
            let iters = (4_000_000 / n).clamp(3, 150) as u32;

            let expect = legacy(text);
            let exact = simd(text).as_bytes() == expect.as_bytes()
                && sclr(text).as_bytes() == expect.as_bytes();

            let t_simd = best(n, iters, || {
                black_box(simd(black_box(text)));
            });
            let t_sclr = best(n, iters, || {
                black_box(sclr(black_box(text)));
            });
            let t_leg = best(n, iters, || {
                black_box(legacy(black_box(text)));
            });
            println!(
                "{name:<6} {label:<9} {n:>7} {t_simd:>7.3} {t_sclr:>7.3} {t_leg:>8.3} | {:>6.1}x {:>6}",
                t_leg / t_simd,
                if exact { "✓" } else { "✗" }
            );
        }
    }
}
