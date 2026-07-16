//! The full normalizer benchmark: every tk-encode pipeline normalizer (the shipping path, now
//! atomnorm-backed) against **tokenizers v0.23.1** — the published crate, driven through its real
//! `NormalizedString` path — on the Wikipedia corpora.
//!
//! run: cargo bench -p tk-encode --bench normalize --features bench-baseline
//!
//! `exact` compares output bytes with v0.23.1. Note the release crate normalizes with
//! `unicode-normalization-alignments` (an older Unicode snapshot), so a ✗ on exotic codepoints
//! reflects Unicode-version drift in the baseline, not a pipeline bug.
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;
use tk_encode::normalizers::{
    BertNormalizer, Lowercase, Nmt, Precompiled, StripAccents, NFC, NFD, NFKC, NFKD,
};
use tk_encode::pipeline::Normalizer;
use tokenizers_release::tokenizer::{NormalizedString, Normalizer as _};
use tokenizers_release::NormalizerWrapper as Legacy;

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

/// The albert nmt_nfkc charsmap, deserialized for either crate's `Precompiled`.
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

    type New<'x> = Box<dyn Fn(&str) -> Cow<'_, str> + 'x>;
    let rows: Vec<(&str, New, Legacy)> = vec![
        (
            "NFC",
            Box::new(|s| Normalizer::normalize(&NFC, s).unwrap()),
            tokenizers_release::normalizers::NFC.into(),
        ),
        (
            "NFD",
            Box::new(|s| Normalizer::normalize(&NFD, s).unwrap()),
            tokenizers_release::normalizers::NFD.into(),
        ),
        (
            "NFKC",
            Box::new(|s| Normalizer::normalize(&NFKC, s).unwrap()),
            tokenizers_release::normalizers::NFKC.into(),
        ),
        (
            "NFKD",
            Box::new(|s| Normalizer::normalize(&NFKD, s).unwrap()),
            tokenizers_release::normalizers::NFKD.into(),
        ),
        (
            "lower",
            Box::new(|s| Normalizer::normalize(&Lowercase, s).unwrap()),
            tokenizers_release::normalizers::Lowercase.into(),
        ),
        (
            "strip",
            Box::new(|s| Normalizer::normalize(&StripAccents, s).unwrap()),
            tokenizers_release::normalizers::StripAccents.into(),
        ),
        (
            "nmt",
            Box::new(|s| Normalizer::normalize(&Nmt, s).unwrap()),
            tokenizers_release::normalizers::Nmt.into(),
        ),
        (
            "bert",
            Box::new(move |s| Normalizer::normalize(&bert, s).unwrap()),
            tokenizers_release::normalizers::BertNormalizer::default().into(),
        ),
        (
            "spm",
            Box::new(move |s| Normalizer::normalize(&precompiled, s).unwrap()),
            albert_precompiled::<tokenizers_release::normalizers::Precompiled>().into(),
        ),
    ];

    println!(
        "\nns/byte, lower=better. new = the shipping tk-encode pipeline normalizer (atomnorm-backed);\n\
         v0.23.1 = the published tokenizers crate through its NormalizedString path. exact = byte-identical\n\
         output. bert = default config; spm = albert nmt_nfkc charsmap.\n"
    );
    println!(
        "{:<6} {:<9} {:>7} {:>7} {:>8} | {:>8} {:>6}",
        "norm", "lang", "bytes", "new", "v0.23.1", "vs0.23", "exact"
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

            let run_legacy = |s: &str| -> NormalizedString {
                let mut ns = NormalizedString::from(s);
                legacy.normalize(&mut ns).unwrap();
                ns
            };
            let exact = new(text).as_bytes() == run_legacy(text).get().as_bytes();
            let t_new = best(n, iters, || {
                black_box(new(black_box(text)));
            });
            let t_leg = best(n, iters, || {
                black_box(run_legacy(black_box(text)));
            });
            println!(
                "{name:<6} {label:<9} {n:>7} {t_new:>7.3} {t_leg:>8.3} | {:>7.1}x {:>6}",
                t_leg / t_new,
                if exact { "✓" } else { "✗" }
            );
        }
        println!();
    }
}
