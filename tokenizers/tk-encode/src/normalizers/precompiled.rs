//! SentencePiece `Precompiled` charsmap normalizer, with an atomnorm skip-scan prefilter.
//!
//! The exact algorithm (grapheme-walk every ≤6-byte grapheme through the charsmap trie) costs
//! 9–18 ns/B — 52–85% of it grapheme segmentation alone. Real text almost never matches, so at
//! model load we derive from the trie the **scan-hot set** (single-char keys ∪ multi-char-key
//! tails: every matchable grapheme contains one) and skip cold runs with [`atomnorm::Scanner`];
//! at a hit we walk back over **cluster-class** chars (can be a non-first char of a grapheme,
//! probed from `unicode_segmentation` itself) to a provable grapheme boundary and run the exact
//! walk locally. Byte-exact by construction, verified per charsmap: if any multi-char key's tail
//! chars are not all cluster-class (the reachability invariant), the prefilter is dropped and the
//! plain walk runs. ~1 ns/B on Wikipedia corpora (10–19× over the plain walk).
use std::borrow::Cow;
use std::convert::TryInto;
use std::sync::OnceLock;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use unicode_segmentation::UnicodeSegmentation;

/// The tk `Precompiled` normalizer: the exact `spm_precompiled` engine plus a lazily-built,
/// per-charsmap scan prefilter. Serialization is transparent (identical to `spm_precompiled`).
#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Precompiled {
    inner: spm_precompiled::Precompiled,
    #[serde(skip)]
    prefilter: OnceLock<Option<atomnorm::Scanner>>,
}

impl Clone for Precompiled {
    fn clone(&self) -> Self {
        Precompiled {
            inner: self.inner.clone(),
            prefilter: OnceLock::new(), // rebuilt on demand
        }
    }
}

impl PartialEq for Precompiled {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

// ── prefilter construction ────────────────────────────────────────────────────────────────────────

/// Chars that can be a NON-FIRST char of a grapheme cluster, probed against the very
/// `unicode_segmentation` that does the exact walk (so the two can never disagree). Astral chars
/// are conservatively class. Charsmap-independent: built once per process (~ms).
fn class_set() -> &'static [u64; 1024] {
    static S: OnceLock<Box<[u64; 1024]>> = OnceLock::new();
    S.get_or_init(|| {
        let mut set = Box::new([0u64; 1024]);
        // letter, CR (CRLF), regional indicator, jamo L, hangul syllable
        let bases = ["a", "\r", "\u{1F1E6}", "\u{1100}", "\u{AC00}"];
        let mut buf = String::new();
        for cp in 1u32..0x10000 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let mut cluster = bases.iter().any(|b| {
                buf.clear();
                buf.push_str(b);
                buf.push(c);
                buf.graphemes(true).count() == 1
            });
            if !cluster {
                // prepend-class: joins with a FOLLOWING char
                buf.clear();
                buf.push(c);
                buf.push('a');
                cluster = buf.graphemes(true).count() == 1;
            }
            if cluster {
                set[(cp >> 6) as usize] |= 1 << (cp & 63);
            }
        }
        set
    })
}

#[inline]
fn class_hot(set: &[u64; 1024], cp: u32) -> bool {
    cp >= 0x10000 || set[(cp >> 6) as usize] >> (cp & 63) & 1 != 0
}

/// DoubleArray key enumeration over the raw charsmap blob (same unit encoding `spm_precompiled`
/// reads). Every access is bounds-checked; `None` on anything malformed → caller falls back.
struct RawTrie(Vec<u32>);

impl RawTrie {
    fn parse(blob: &[u8]) -> Option<RawTrie> {
        let trie_size = u32::from_le_bytes(blob.get(0..4)?.try_into().ok()?) as usize;
        let n = trie_size / 4;
        let mut array = Vec::with_capacity(n);
        for k in 0..n {
            let off = 4 + 4 * k;
            array.push(u32::from_le_bytes(blob.get(off..off + 4)?.try_into().ok()?));
        }
        Some(RawTrie(array))
    }
    fn offset(u: u32) -> usize {
        ((u as usize) >> 10) << (((u as usize) & (1 << 9)) >> 6)
    }
    /// All keys of ≤ `max_len` bytes (the walk only probes graphemes < 6 bytes).
    fn keys(&self, max_len: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        let root = *self.0.first().unwrap_or(&0);
        self.dfs(Self::offset(root), &mut prefix, max_len, &mut out);
        out
    }
    fn dfs(&self, pos: usize, prefix: &mut Vec<u8>, max_len: usize, out: &mut Vec<Vec<u8>>) {
        if prefix.len() >= max_len {
            return;
        }
        for c in 1usize..=255 {
            let child = pos ^ c;
            let Some(&unit) = self.0.get(child) else {
                continue;
            };
            if (unit as usize) & ((1 << 31) | 0xFF) != c {
                continue; // label mismatch: no such edge
            }
            prefix.push(c as u8);
            if (unit >> 8) & 1 == 1 {
                out.push(prefix.clone());
            }
            self.dfs(child ^ Self::offset(unit), prefix, max_len, out);
            prefix.pop();
        }
    }
}

/// Build the scan-hot set and verify the reachability invariant. `None` = use the plain walk.
fn build_prefilter(inner: &spm_precompiled::Precompiled) -> Option<atomnorm::Scanner> {
    // single-char keys: probe the public API for every BMP cp
    let mut hot = Box::new([0u64; 1024]);
    let mut buf = [0u8; 4];
    for cp in 1u32..0x10000 {
        if let Some(c) = char::from_u32(cp) {
            if inner.transform(c.encode_utf8(&mut buf)).is_some() {
                hot[(cp >> 6) as usize] |= 1 << (cp & 63);
            }
        }
    }
    // multi-char keys need the trie: recover the blob through the (private-field) serde form
    let blob_b64 = serde_json::to_value(inner).ok()?;
    let blob = b64_decode(blob_b64.get("precompiled_charsmap")?.as_str()?);
    let keys = RawTrie::parse(&blob)?.keys(5);
    let class = class_set();
    for k in &keys {
        let Ok(s) = std::str::from_utf8(k) else {
            continue; // non-UTF-8 key: unreachable from &str graphemes
        };
        for c in s.chars().skip(1) {
            // reachability invariant: a grapheme matching a multi-char key must contain a hot
            // char even when its first char is cold — i.e. every tail char is cluster-class
            // (the backup walk re-derives the boundary) AND hot (the scan sees it)
            if !class_hot(class, c as u32) {
                return None;
            }
            let cp = c as u32;
            if cp < 0x10000 {
                hot[(cp >> 6) as usize] |= 1 << (cp & 63);
            }
        }
    }
    Some(atomnorm::Scanner::new(&hot, true)) // astral: conservatively always hot
}

/// Minimal standard-alphabet base64 (the charsmap round-trips through serde as base64).
fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| -> i32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as i32,
            b'a'..=b'z' => (c - b'a' + 26) as i32,
            b'0'..=b'9' => (c - b'0' + 52) as i32,
            b'+' => 62,
            b'/' => 63,
            _ => -1,
        }
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut nbits) = (0u32, 0u32);
    for &b in s.as_bytes() {
        let v = val(b);
        if v < 0 {
            continue;
        }
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    out
}

// ── normalization ─────────────────────────────────────────────────────────────────────────────────

impl Precompiled {
    /// The prefiltered pipeline path: skip cold runs, and at each hot char walk back over
    /// cluster-class chars to a provable grapheme boundary, then run the exact walk until the
    /// next grapheme starts cold again.
    fn normalize_scan<'a>(&self, scanner: &atomnorm::Scanner, input: &'a str) -> Cow<'a, str> {
        let bytes = input.as_bytes();
        let n = bytes.len();
        let class = class_set();
        let mut out: Option<String> = None;
        let mut verb = 0usize; // start of the pending verbatim run
        let mut i = 0usize;
        while i < n {
            i = scanner.next_member(input, i);
            if i >= n {
                break;
            }
            // back up over class chars, then one more: that char starts its own grapheme
            let mut p = i;
            while p > verb {
                let mut q = p - 1;
                while q > verb && bytes[q] & 0xC0 == 0x80 {
                    q -= 1;
                }
                let qc = input[q..].chars().next().unwrap();
                p = q;
                if !class_hot(class, qc as u32) {
                    break;
                }
            }
            if let Some(o) = out.as_mut() {
                o.push_str(&input[verb..p]);
            }
            // exact walk from the boundary; leave when the next grapheme starts cold
            let mut exit = n;
            for (off, grapheme) in input[p..].grapheme_indices(true) {
                let abs = p + off;
                if abs > i && !scanner.contains(input[abs..].chars().next().unwrap()) {
                    exit = abs;
                    break;
                }
                let mut done = false;
                if grapheme.len() < 6 {
                    if let Some(rep) = self.inner.transform(grapheme) {
                        let o = out.get_or_insert_with(|| {
                            let mut s = String::with_capacity(input.len());
                            s.push_str(&input[..abs]);
                            s
                        });
                        o.push_str(rep);
                        done = true;
                    }
                }
                if !done {
                    for (ci, c) in grapheme.char_indices() {
                        if let Some(rep) = self.inner.transform(&grapheme[ci..ci + c.len_utf8()]) {
                            let o = out.get_or_insert_with(|| {
                                let mut s = String::with_capacity(input.len());
                                s.push_str(&input[..abs + ci]);
                                s
                            });
                            o.push_str(rep);
                        } else if let Some(o) = out.as_mut() {
                            o.push(c);
                        }
                    }
                }
            }
            verb = exit;
            i = exit;
        }
        match out {
            Some(mut s) => {
                s.push_str(&input[verb..]);
                Cow::Owned(s)
            }
            None => Cow::Borrowed(input),
        }
    }

    /// The plain exact walk — the fallback when the invariant fails, and the test oracle.
    fn normalize_walk<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut transformed: Option<String> = None;
        for (g_idx, grapheme) in input.grapheme_indices(true) {
            if grapheme.len() < 6 {
                if let Some(replacement) = self.inner.transform(grapheme) {
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
                if let Some(replacement) = self
                    .inner
                    .transform(&grapheme[c_idx..c_idx + character.len_utf8()])
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
}

impl pipeline::Normalizer for Precompiled {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        match self.prefilter.get_or_init(|| build_prefilter(&self.inner)) {
            Some(scanner) => Ok(self.normalize_scan(scanner, input)),
            None => Ok(self.normalize_walk(input)),
        }
    }
}

// ── legacy NormalizedString path (unchanged behavior) ─────────────────────────────────────────────

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
                if let Some(norm) = self.inner.transform(grapheme) {
                    modified = true;
                    replace(&mut transformations, grapheme, norm);
                    return;
                }
            }
            for (char_index, c) in grapheme.char_indices() {
                let part = &grapheme[char_index..char_index + c.len_utf8()];
                if let Some(norm) = self.inner.transform(part) {
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

    const INPUTS: &[&str] = &[
        "™\x1eg",
        "ＫＡＤＯＫＡＷＡ",
        "１２３",
        "…",
        "\u{fb01}",
        "e\u{0301}",
        "㍿",
        "abc def",
        "",
        "a\u{0301}\u{0323}x",
        "\u{1100}\u{1161}\u{11A8}",
        " \u{0301}",
        "🇫🇷🇩🇪 flags",
        "café \r\n straße",
        "ﬁ\u{200d}x",
        "中文，标点！Ｆｕｌｌ width",
    ];

    #[test]
    fn pipeline_precompiled_matches_legacy() {
        let n = albert_precompiled();
        let mut any_modified = false;
        for input in INPUTS {
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
    fn prefilter_builds_and_matches_plain_walk() {
        let n = albert_precompiled();
        // the invariant must hold for the standard nmt_nfkc charsmap — otherwise the fast path
        // silently degrades to the plain walk
        let scanner = n
            .prefilter
            .get_or_init(|| build_prefilter(&n.inner))
            .as_ref()
            .expect("albert charsmap should satisfy the prefilter invariant");
        for input in INPUTS {
            let walk = n.normalize_walk(input);
            let scan = n.normalize_scan(scanner, input);
            assert_eq!(walk, scan, "scan diverges from walk on {input:?}");
            assert_eq!(
                matches!(walk, Cow::Borrowed(_)),
                matches!(scan, Cow::Borrowed(_)),
                "borrow parity on {input:?}"
            );
        }
    }

    #[test]
    #[ignore = "perf check on the Wikipedia corpora — run with --release"]
    fn prefilter_corpus_throughput() {
        let n = albert_precompiled();
        let scanner = n
            .prefilter
            .get_or_init(|| build_prefilter(&n.inner))
            .as_ref()
            .unwrap();
        for rel in [
            "../data/big.txt",
            "../atomsplit/benches/data/ru.txt",
            "../atomsplit/benches/data/hi.txt",
            "../atomsplit/benches/data/zh.txt",
            "../atomsplit/benches/data/ko.txt",
        ] {
            let Ok(s) = std::fs::read_to_string(rel) else {
                continue;
            };
            let mut c = s.len().min(180_000);
            while c > 0 && !s.is_char_boundary(c) {
                c -= 1;
            }
            let text = &s[..c];
            assert_eq!(n.normalize_walk(text), n.normalize_scan(scanner, text));
            let best = |f: &dyn Fn() -> usize| {
                let mut b = f64::INFINITY;
                for _ in 0..5 {
                    let t = std::time::Instant::now();
                    for _ in 0..8 {
                        std::hint::black_box(f());
                    }
                    b = b.min(t.elapsed().as_nanos() as f64 / (8 * text.len()) as f64);
                }
                b
            };
            let t_walk = best(&|| n.normalize_walk(text).len());
            let t_scan = best(&|| n.normalize_scan(scanner, text).len());
            eprintln!(
                "{rel:<40} walk {t_walk:>6.2} ns/B  scan {t_scan:>5.2} ns/B  {:>5.1}x",
                t_walk / t_scan
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
