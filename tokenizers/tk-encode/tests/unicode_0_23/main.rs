//! `Lowercase` and `Digits` must match the released 0.23 wheels over the whole code space.
//!
//! In 0.23 both took their Unicode data from std, so the released crate is no oracle here: built
//! today, it follows today's toolchain. The wheels were built with rustc 1.92 to 1.98, all on
//! Unicode 17. `blocks.txt` holds what they output, hashed per block of 256 code points, and
//! `generate.py` rebuilds it. Nothing here depends on the toolchain that runs the test.

use std::collections::{BTreeMap, BTreeSet};

use tk_encode::normalizers::utils::Lowercase;
use tk_encode::pipeline::{Normalize, PreTokenize, PreTokenizerScratch, Span};
use tk_encode::pre_tokenizers::digits::Digits;

const BLOCKS: &str = include_str!("blocks.txt");
const BLOCK: u32 = 256;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;
const PIECE_END: u8 = 0xFE;
const OUTPUT_END: u8 = 0xFF;

/// Hashes every block like `generate.py` and compares with its lines for `component`. `run` appends
/// the output for one code point and says whether it left the input unchanged.
fn assert_matches_wheel(component: &str, mut run: impl FnMut(char, &mut Vec<u8>) -> bool) {
    let expected: BTreeMap<u32, u64> = BLOCKS
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut fields = line.split(' ');
            (fields.next() == Some(component)).then(|| {
                let start = u32::from_str_radix(fields.next().unwrap(), 16).unwrap();
                let digest = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
                (start, digest)
            })
        })
        .collect();
    assert!(!expected.is_empty(), "no `{component}` lines in blocks.txt");

    let mut got = BTreeMap::new();
    let mut output = Vec::new();
    for start in (0..=char::MAX as u32).step_by(BLOCK as usize) {
        let (mut digest, mut changed) = (FNV_OFFSET, false);
        for c in (start..start + BLOCK).filter_map(char::from_u32) {
            output.clear();
            changed |= !run(c, &mut output);
            output.push(OUTPUT_END);
            for &byte in &output {
                digest = (digest ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
            }
        }
        if changed {
            got.insert(start, digest);
        }
    }

    let starts: BTreeSet<&u32> = expected.keys().chain(got.keys()).collect();
    let moved: Vec<String> = starts
        .into_iter()
        .filter(|start| expected.get(start) != got.get(start))
        .map(|&start| format!("U+{start:04X}..U+{:04X}", start + BLOCK - 1))
        .collect();
    assert!(
        moved.is_empty(),
        "{component}: {} blocks differ from the 0.23 wheels: {moved:?}",
        moved.len()
    );
}

fn lowercase(normalizer: &dyn Normalize) -> impl FnMut(char, &mut Vec<u8>) -> bool {
    move |c, output| {
        let input = c.to_string();
        let lowered = normalizer.normalize(&input, 0).unwrap();
        output.extend_from_slice(lowered.as_bytes());
        *lowered == input
    }
}

fn digits(individual_digits: bool) -> impl FnMut(char, &mut Vec<u8>) -> bool {
    let digits = Digits::new(individual_digits);
    let mut scratch = PreTokenizerScratch::default();
    let mut spans: Vec<Span> = Vec::new();
    move |c, output| {
        // Two in a row so a numeric code point splits in both modes.
        let text = format!("a{c}{c}a");
        spans.clear();
        digits
            .pre_tokenize(&text, &mut scratch, &mut spans)
            .unwrap();
        for span in &spans {
            output.extend_from_slice(text[span.range()].as_bytes());
            output.push(PIECE_END);
        }
        spans.len() == 1
    }
}

#[test]
fn lowercase_matches_wheel() {
    assert_matches_wheel("lowercase", lowercase(&Lowercase));
}

/// 0.23's `BertNormalizer` lowercased like `Lowercase`, and `generate.py` checks the wheel agrees.
#[cfg(feature = "normalizers")]
#[test]
fn bert_lowercase_matches_wheel() {
    let bert = tk_encode::normalizers::BertNormalizer::new(false, false, Some(false), true);
    assert_matches_wheel("lowercase", lowercase(&bert));
}

#[test]
fn digits_matches_wheel() {
    assert_matches_wheel("digits", digits(false));
}

#[test]
fn individual_digits_matches_wheel() {
    assert_matches_wheel("digits_individual", digits(true));
}
