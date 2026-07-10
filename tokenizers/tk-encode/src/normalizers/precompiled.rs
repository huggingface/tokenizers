use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
pub use spm_precompiled::Precompiled;
use std::cmp::Ordering;
use unicode_normalization_alignments::char::is_combining_mark;
use unicode_segmentation::UnicodeSegmentation;

/// Which codepoints the charsmap actually rewrites as a *lone* codepoint (bit set = non-inert). Built by
/// probing `transform` over every codepoint once — so it's byte-exact for ANY charsmap, not just the
/// canonical `nmt_nfkc`. Only ~4.8k of 1.1M are non-inert, so the bulk of text is a copy.
struct NonInert(Box<[u64]>);

impl NonInert {
    fn build(p: &Precompiled) -> Self {
        let mut bits = vec![0u64; 0x110000usize.div_ceil(64)].into_boxed_slice();
        let mut buf = [0u8; 4];
        for cp in 0..0x110000u32 {
            if let Some(c) = char::from_u32(cp) {
                if p.transform(c.encode_utf8(&mut buf)).is_some() {
                    bits[(cp >> 6) as usize] |= 1u64 << (cp & 63);
                }
            }
        }
        NonInert(bits)
    }
    #[inline]
    fn non_inert(&self, c: char) -> bool {
        let cp = c as u32;
        (self.0[(cp >> 6) as usize] >> (cp & 63)) & 1 != 0
    }
}

/// Behavioral fingerprint over a fixed set of codepoints the charsmaps in the wild differ on — enough to
/// tell one charsmap from another cheaply (8 probes) so the per-thread cache can key on it.
fn fingerprint(p: &Precompiled) -> u64 {
    const PROBES: [char; 8] = [
        '\u{FF21}', '\u{2460}', '\u{FB01}', '\u{3000}', '\u{00A0}', '\u{2126}', '\u{FE30}',
        '\u{017F}',
    ];
    let mut h = 0xcbf29ce484222325u64;
    let mut buf = [0u8; 4];
    for &c in &PROBES {
        for b in p
            .transform(c.encode_utf8(&mut buf))
            .unwrap_or("\u{1}")
            .bytes()
        {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        h = (h ^ 0xff).wrapping_mul(0x100000001b3);
    }
    h
}

thread_local! {
    /// One entry per thread — a tokenizer uses a single charsmap, so this hits every call after the
    /// first; the (expensive) full-codepoint probe runs once. Keyed by fingerprint so a second charsmap
    /// on the same thread rebuilds correctly.
    static NON_INERT: RefCell<Option<(u64, Rc<NonInert>)>> = const { RefCell::new(None) };
}

fn non_inert_for(p: &Precompiled) -> Rc<NonInert> {
    let fp = fingerprint(p);
    NON_INERT.with(|cell| {
        if let Some((cfp, arc)) = cell.borrow().as_ref() {
            if *cfp == fp {
                return arc.clone();
            }
        }
        let arc = Rc::new(NonInert::build(p));
        *cell.borrow_mut() = Some((fp, arc.clone()));
        arc
    })
}

/// The reference per-grapheme rewrite: try the whole grapheme (charsmap rules are < 6 bytes), else fall
/// back per-char. Identical to the legacy/general path — only invoked on the rare non-inert graphemes.
#[inline]
fn apply_grapheme(p: &Precompiled, g: &str, out: &mut String) {
    if g.len() < 6 {
        if let Some(r) = p.transform(g) {
            out.push_str(r);
            return;
        }
    }
    for (ci, c) in g.char_indices() {
        match p.transform(&g[ci..ci + c.len_utf8()]) {
            Some(r) => out.push_str(r),
            None => out.push(c),
        }
    }
}

fn replace(transformations: &mut Vec<(char, isize)>, old_part: &str, new_part: &str) {
    let old_count = old_part.chars().count() as isize;
    let new_count = new_part.chars().count() as isize;
    let diff = new_count - old_count;

    // If we are just replacing characters, all changes should be == 0
    transformations.extend(new_part.chars().map(|c| (c, 0)));

    match diff.cmp(&0) {
        // If we are adding some characters, the last DIFF characters should be == 1
        Ordering::Greater => {
            transformations
                .iter_mut()
                .rev()
                .take(diff as usize)
                .for_each(|(_, cs)| *cs = 1);
        }
        // If we are removing some characters, the last one should include the diff
        Ordering::Less => {
            if let Some((_, cs)) = transformations.last_mut() {
                *cs += diff;
            }
        }
        _ => {}
    }
}

impl Normalizer for Precompiled {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut transformations = Vec::with_capacity(normalized.get().len());
        // Future reader. From @Narsil.
        // Yes, this is weird,
        // Yes, this seems broken
        // No, I don't know why Google did this.
        // If you question this code, check this normalizer against
        // XNLI database (all languages) with Unigram model against
        // Mbart, XLMRoberta *AND* Marian. If you don't get 100% or
        // break a single test.
        // You don't pass.
        let mut modified = false;
        normalized.get().graphemes(true).for_each(|grapheme| {
            if grapheme.len() < 6 {
                if let Some(norm) = self.transform(grapheme) {
                    modified = true;
                    replace(&mut transformations, grapheme, norm);
                    return;
                }
            }
            for (char_index, c) in grapheme.char_indices() {
                let part = &grapheme[char_index..char_index + c.len_utf8()];
                if let Some(norm) = self.transform(part) {
                    modified = true;
                    replace(&mut transformations, part, norm);
                } else {
                    transformations.push((c, 0));
                }
            }
        });
        if modified {
            normalized.transform(transformations, 0);
        }
        Ok(())
    }
}

impl pipeline::Normalizer for Precompiled {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        let ni = non_inert_for(self);
        let bytes = input.as_bytes();
        let n = bytes.len();
        let mut out: Option<String> = None; // None ⇒ still borrowable
        let mut i = 0;
        while i < n {
            // Fast inert run: codepoints the charsmap leaves alone that also can't be a composition base.
            // A combining mark (or a char immediately followed by one) forms a multi-codepoint grapheme,
            // which is where the charsmap's only < 6-byte whole-grapheme rules (canonical composition)
            // live — so break there and let `graphemes(true)` segment it exactly.
            let run_start = i;
            while i < n {
                let c = input[i..].chars().next().unwrap();
                if ni.non_inert(c) || is_combining_mark(c) {
                    break;
                }
                let j = i + c.len_utf8();
                if j < n && is_combining_mark(input[j..].chars().next().unwrap()) {
                    break;
                }
                i = j;
            }
            if let Some(s) = out.as_mut() {
                s.push_str(&input[run_start..i]); // once owning, copy the inert run
            }
            if i >= n {
                break;
            }
            // Non-inert grapheme → the exact per-grapheme rewrite.
            let g = input[i..].graphemes(true).next().unwrap();
            let s = out.get_or_insert_with(|| {
                let mut s = String::with_capacity(n);
                s.push_str(&input[..i]); // borrowed inert prefix, materialized once
                s
            });
            apply_grapheme(self, g, s);
            i += g.len();
        }
        Ok(match out {
            Some(s) => Cow::Owned(s),
            None => Cow::Borrowed(input),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn albert_precompiled() -> Precompiled {
        let json = std::fs::read_to_string("../data/albert-base-v1-tokenizer.json").unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let precompiled = value["normalizer"]["normalizers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["type"] == "Precompiled")
            .unwrap();
        // Precompiled can't deserialize through serde_json::Value (the base64
        // charsmap only decodes via the string deserializer) — same dance as
        // NormalizerWrapper's Deserialize impl
        serde_json::from_str(&serde_json::to_string(precompiled).unwrap()).unwrap()
    }

    #[test]
    fn pipeline_precompiled_matches_legacy() {
        let n = albert_precompiled();
        let mut any_modified = false;
        for input in &[
            "™\x1eg",
            "ＫＡＤＯＫＡＷＡ",
            "１２３",
            "…",
            "\u{fb01}",
            "e\u{0301}",
            "㍿",
            "abc def",
            "",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            any_modified |= ns.get() != *input;
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                "pipeline output diverges from legacy for {input:?}"
            );
        }
        // Guard against the oracle silently becoming a no-op on these inputs
        assert!(any_modified);
    }

    #[test]
    fn pipeline_precompiled_matches_legacy_corpus() {
        // Strong byte-exactness check for the inert-run fast path: a big mixed corpus exercising inert
        // runs, non-inert graphemes, combining-mark composition boundaries, CJK, fullwidth, ZWJ/emoji,
        // exotic spaces, and control chars — the segmentation/boundary edge cases the fast path skips.
        let n = albert_precompiled();
        let corpus = [
            "The quick brown fox JUMPS over the lazy dog 12345.",
            "Съешь же ещё этих мягких французских булок да выпей чаю.",
            "这是一个包含若干汉字的中文测试句子，还有标点。",
            "ＫＡＤＯＫＡＷＡ　１２３４５　㍿　…　½　Ⅳ",
            "café déjà Å ﬁ ﬂ œuvre e\u{0301}a\u{0300}o\u{0323}\u{0301}",
            "नमस्ते दुनिया विश्व",
            "한국어 안녕하세요 테스트",
            "emoji 👨‍👩‍👧‍👦 🇫🇷 café\u{200d}x tab\there\nnewline\u{00a0}nbsp\u{3000}ideo",
            "Ω ohm \u{2126} and \u{017f}tretch mixed 中x文a य़ ",
        ]
        .join("  ");
        // repeat with offsets so runs start/end at varied boundaries
        for skip in 0..4 {
            let input = &corpus[corpus.char_indices().nth(skip).map(|(i, _)| i).unwrap_or(0)..];
            let mut ns = NormalizedString::from(input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                "fast path diverges from legacy (skip={skip})"
            );
        }
    }

    #[test]
    fn expansion_followed_by_removal() {
        // Simulate transformations from "™\x1eg" to "TMg"
        let mut transformations = vec![];

        let mut n = NormalizedString::from("™\x1eg");
        replace(&mut transformations, "™", "TM");
        replace(&mut transformations, "\x1e", "");
        transformations.push(('g', 0));

        n.transform(transformations, 0);

        assert_eq!(n.get(), "TMg");
    }
}
