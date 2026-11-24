//! Unicode normalization must match the latest *released* `tokenizers` over the whole code space.
//!
//! 0.23 normalizes with the Unicode 9 tables of `unicode-normalization-alignments`, compiled into the
//! crate, so the release is an oracle whose answers do not depend on the toolchain. Every code point
//! runs alone, doubled, next to marks that expose its combining class, and decomposed by the current
//! tables, which spells out every composition either side knows.
//!
//!   make oracle

#![cfg(feature = "bench-baseline")]

use std::sync::OnceLock;

use tk_encode::normalizers::{BertNormalizer, NFC, NFD, NFKC, NFKD, StripAccents};
use tk_encode::pipeline::Normalize;
use tokenizers_release::normalizers as release;
use tokenizers_release::{NormalizedString, Normalizer};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_public_assigned;

/// Code points per batch: the release tracks an alignment per byte, so one batch for the whole code
/// space would cost gigabytes.
const BATCH: u32 = 0x1000;

/// Every input for the code points of `range`, one per line. `\n` is a starter nothing composes with,
/// so the lines normalize independently.
fn batch(range: std::ops::Range<u32>) -> String {
    let mut text = String::new();
    for c in range.filter_map(char::from_u32) {
        text.push(c);
        text.push('\n');
        // Unassigned now means unassigned in Unicode 9 too: nothing to compose or reorder.
        if !is_public_assigned(c) {
            continue;
        }
        let decomposed: String = std::iter::once(c).nfd().collect();
        let compat: String = std::iter::once(c).nfkd().collect();
        // U+0334 has the lowest non-zero class and U+0301 one of the highest, so a mark of any class
        // between them gets reordered.
        for line in [
            format!("{c}{c}"),
            format!("a{c}\u{334}"),
            format!("a\u{301}{c}"),
        ] {
            text.push_str(&line);
            text.push('\n');
        }
        for line in [decomposed, compat] {
            if line.chars().ne(std::iter::once(c)) {
                text.push_str(&line);
                text.push('\n');
            }
        }
    }
    text
}

fn batches() -> &'static [String] {
    static BATCHES: OnceLock<Vec<String>> = OnceLock::new();
    BATCHES.get_or_init(|| {
        (0..=char::MAX as u32)
            .step_by(BATCH as usize)
            .map(|start| batch(start..start + BATCH))
            .collect()
    })
}

fn hex(s: &str) -> String {
    let cps: Vec<String> = s.chars().map(|c| format!("{:04X}", c as u32)).collect();
    cps.join(" ")
}

fn assert_matches_release(name: &str, ours: &dyn Normalize, theirs: &dyn Normalizer) {
    let mut moved = Vec::new();
    for text in batches() {
        let got = ours.normalize(text, 0).unwrap();
        let mut expected = NormalizedString::from(text.as_str());
        theirs.normalize(&mut expected).unwrap();
        if *got == *expected.get() {
            continue;
        }
        for ((input, got), expected) in text
            .split('\n')
            .zip(got.split('\n'))
            .zip(expected.get().split('\n'))
        {
            if got != expected {
                moved.push(format!(
                    "[{}] -> [{}], 0.23 gives [{}]",
                    hex(input),
                    hex(got),
                    hex(expected)
                ));
            }
        }
    }
    assert!(
        moved.is_empty(),
        "{name}: {} inputs differ from 0.23, first: {:#?}",
        moved.len(),
        &moved[..moved.len().min(20)]
    );
}

#[test]
fn nfc() {
    assert_matches_release("NFC", &NFC, &release::NFC);
}

#[test]
fn nfd() {
    assert_matches_release("NFD", &NFD, &release::NFD);
}

#[test]
fn nfkc() {
    assert_matches_release("NFKC", &NFKC, &release::NFKC);
}

#[test]
fn nfkd() {
    assert_matches_release("NFKD", &NFKD, &release::NFKD);
}

#[test]
fn strip_accents() {
    assert_matches_release("StripAccents", &StripAccents::new(), &release::StripAccents);
}

/// Accent stripping alone: cleaning and lowercasing take their data from std, which follows the
/// toolchain on both sides.
#[test]
fn bert_strip_accents() {
    assert_matches_release(
        "BertNormalizer",
        &BertNormalizer::new(false, false, Some(true), false),
        &release::BertNormalizer::new(false, false, Some(true), false),
    );
}
