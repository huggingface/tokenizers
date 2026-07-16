//! The full normalizer benchmark: every tk-encode pipeline normalizer (the real shipping path,
//! now atomnorm-backed) against its LEGACY implementation — the char-iterator / grapheme-walk code
//! each one ran before — on the Wikipedia corpora.
//! run: cargo bench -p tk-encode --bench normalize
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;
use tk_encode::normalizers::{
    BertNormalizer, Lowercase, Nmt, Precompiled, StripAccents, NFC, NFD, NFKC, NFKD,
};
use tk_encode::pipeline::Normalizer;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

// ── legacy reference implementations (what each normalizer ran before atomnorm) ──────────────────

fn leg_form<'a>(form: &str, s: &'a str) -> Cow<'a, str> {
    Cow::Owned(match form {
        "NFC" => s.nfc().collect(),
        "NFD" => s.nfd().collect(),
        "NFKC" => s.nfkc().collect(),
        _ => s.nfkd().collect(),
    })
}

fn lowercases_to_self(c: char) -> bool {
    let mut it = c.to_lowercase();
    matches!((it.next(), it.next()), (Some(first), None) if first == c)
}

fn leg_lower(input: &str) -> Cow<'_, str> {
    if input.chars().all(lowercases_to_self) {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.chars().flat_map(|c| c.to_lowercase()).collect())
    }
}

fn leg_strip_accents(input: &str) -> Cow<'_, str> {
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
        0x0009 | 0x000A | 0x000C | 0x000D | 0x1680 | 0x200B..=0x200F | 0x2028 | 0x2029 | 0x2581
        | 0xFEFF | 0xFFFD => ' ',
        _ => c,
    }
}
fn leg_nmt(input: &str) -> Cow<'_, str> {
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

fn b_is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}
fn b_is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}
fn b_removes(c: char) -> bool {
    c == '\0' || c == '\u{fffd}' || b_is_control(c)
}
fn b_is_chinese(c: char) -> bool {
    matches!(c as usize,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F |
        0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}
fn leg_bert(input: &str) -> Cow<'_, str> {
    use unicode_normalization::{is_nfd_quick, IsNormalized};
    let noop = matches!(is_nfd_quick(input.chars()), IsNormalized::Yes)
        && !input.chars().any(|c| {
            b_removes(c)
                || (b_is_whitespace(c) && c != ' ')
                || b_is_chinese(c)
                || c.is_mark_nonspacing()
                || !lowercases_to_self(c)
        });
    if noop {
        return Cow::Borrowed(input);
    }
    let cleaned = input
        .chars()
        .filter(|&c| !b_removes(c))
        .flat_map(|c| {
            let c = if b_is_whitespace(c) { ' ' } else { c };
            if b_is_chinese(c) {
                [Some(' '), Some(c), Some(' ')]
            } else {
                [Some(c), None, None]
            }
        })
        .flatten();
    let mut out = String::with_capacity(input.len());
    out.extend(
        cleaned
            .nfd()
            .filter(|c| !c.is_mark_nonspacing())
            .flat_map(char::to_lowercase),
    );
    Cow::Owned(out)
}

fn leg_precompiled<'a>(pre: &spm_precompiled::Precompiled, input: &'a str) -> Cow<'a, str> {
    let mut transformed: Option<String> = None;
    for (g_idx, grapheme) in input.grapheme_indices(true) {
        if grapheme.len() < 6 {
            if let Some(replacement) = pre.transform(grapheme) {
                let string = transformed.get_or_insert_with(|| {
                    let mut s = String::with_capacity(input.len());
                    s.push_str(&input[..g_idx]);
                    s
                });
                string.push_str(replacement);
                continue;
            }
        }
        for (c_idx, character) in grapheme.char_indices() {
            if let Some(replacement) = pre.transform(&grapheme[c_idx..c_idx + character.len_utf8()])
            {
                let string = transformed.get_or_insert_with(|| {
                    let mut s = String::with_capacity(input.len());
                    s.push_str(&input[..g_idx + c_idx]);
                    s
                });
                string.push_str(replacement);
            } else if let Some(transformed) = transformed.as_mut() {
                transformed.push(character);
            }
        }
    }
    match transformed {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(input),
    }
}

// ── harness ───────────────────────────────────────────────────────────────────────────────────────

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

fn albert_precompiled<T: serde::de::DeserializeOwned>() -> T {
    let json = std::fs::read_to_string("../data/albert-base-v1-tokenizer.json").unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let node = value["normalizer"]["normalizers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["type"] == "Precompiled")
        .unwrap();
    serde_json::from_str(&serde_json::to_string(node).unwrap()).unwrap()
}

fn main() {
    let bert = BertNormalizer::default();
    let precompiled: Precompiled = albert_precompiled();
    let spm: spm_precompiled::Precompiled = albert_precompiled();

    type New<'x> = Box<dyn Fn(&str) -> Cow<'_, str> + 'x>;
    type Leg<'x> = Box<dyn Fn(&str) -> Cow<'_, str> + 'x>;
    let rows: Vec<(&str, New, Leg)> = vec![
        ("NFC", Box::new(|s| Normalizer::normalize(&NFC, s).unwrap()), Box::new(|s| leg_form("NFC", s))),
        ("NFD", Box::new(|s| Normalizer::normalize(&NFD, s).unwrap()), Box::new(|s| leg_form("NFD", s))),
        ("NFKC", Box::new(|s| Normalizer::normalize(&NFKC, s).unwrap()), Box::new(|s| leg_form("NFKC", s))),
        ("NFKD", Box::new(|s| Normalizer::normalize(&NFKD, s).unwrap()), Box::new(|s| leg_form("NFKD", s))),
        ("lower", Box::new(|s| Normalizer::normalize(&Lowercase, s).unwrap()), Box::new(leg_lower)),
        ("strip", Box::new(|s| Normalizer::normalize(&StripAccents, s).unwrap()), Box::new(leg_strip_accents)),
        ("nmt", Box::new(|s| Normalizer::normalize(&Nmt, s).unwrap()), Box::new(leg_nmt)),
        ("bert", Box::new(move |s| Normalizer::normalize(&bert, s).unwrap()), Box::new(leg_bert)),
        ("spm", Box::new(move |s| Normalizer::normalize(&precompiled, s).unwrap()), Box::new(move |s| leg_precompiled(&spm, s))),
    ];

    println!(
        "\nns/byte, lower=better. new = the shipping tk-encode pipeline normalizer (atomnorm-backed);\n\
         legacy = its previous implementation (char iterators / grapheme walk). exact = byte-identical.\n\
         forms compared vs unicode-normalization; bert = default config; spm = albert nmt_nfkc charsmap.\n"
    );
    println!(
        "{:<6} {:<9} {:>7} {:>7} {:>8} | {:>7} {:>6}",
        "norm", "lang", "bytes", "new", "legacy", "vsLeg", "exact"
    );
    for (name, new, legacy) in &rows {
        for (label, rel) in CORPORA {
            let Ok(s) = std::fs::read_to_string(rel) else {
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

            let exact = new(text).as_bytes() == legacy(text).as_bytes();
            let t_new = best(n, iters, || {
                black_box(new(black_box(text)));
            });
            let t_leg = best(n, iters, || {
                black_box(legacy(black_box(text)));
            });
            println!(
                "{name:<6} {label:<9} {n:>7} {t_new:>7.3} {t_leg:>8.3} | {:>6.1}x {:>6}",
                t_leg / t_new,
                if exact { "✓" } else { "✗" }
            );
        }
        println!();
    }
}
