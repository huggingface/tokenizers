use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
pub use spm_precompiled::Precompiled;
use std::sync::atomic::{AtomicU32, Ordering};

use unicode_segmentation::GraphemeCursor;

/// A [`Precompiled`] together with the `precompiled_charsmap` bytes it was parsed from.
///
/// The bytes are kept because `spm_precompiled` holds its `precompiled_charsmap` in a private field
/// and publishes it only through its `Serialize` impl. A serde-free writer therefore has no way to
/// ask the value what it was built from, and this is the one normalizer whose configuration *is*
/// that blob — so either the pipeline remembers it or the normalizer cannot be written back out.
///
/// Remembering it costs a second copy of the map: 237 KB for the SentencePiece charsmap that t5,
/// albert and xlm-roberta all ship. Only a config that has a `Precompiled` pays it, and the copy
/// goes away the day upstream grows a three-line getter.
///
/// [`charsmap`](Self::charsmap) is an `Option` because not every producer has the bytes to hand:
/// one that only has an already-parsed [`Precompiled`] records `None`, and a writer then reports
/// that rather than inventing a blob. `from_charsmap` is currently the only constructor, so today
/// it is always `Some`.
#[derive(Debug, Clone)]
pub struct PrecompiledNormalizer {
    map: Charsmap,
    charsmap: Option<Box<[u8]>>,
}

impl PrecompiledNormalizer {
    /// Parse a charsmap and keep it, which is what a reader does.
    pub fn from_charsmap(charsmap: &[u8]) -> Result<Self> {
        Ok(Self {
            map: Charsmap::parse(charsmap)?,
            charsmap: Some(charsmap.into()),
        })
    }

    /// The bytes this was parsed from, when they are known.
    pub fn charsmap(&self) -> Option<&[u8]> {
        self.charsmap.as_deref()
    }
}

impl pipeline::Normalize for PrecompiledNormalizer {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        Ok(self.map.normalize(input))
    }
}

/// The `precompiled_charsmap` read into the two things a lookup needs, so that normalizing is a
/// walk with no allocation.
///
/// [`spm_precompiled::Precompiled::transform`] returns every match as a fresh `Vec`, keeps the
/// first, and is called once per grapheme *and* again per character of every grapheme it did not
/// match whole. On prose that is an allocation per byte of input. The map is a double array; a
/// lookup is a handful of XORs, and most text needs none at all.
#[derive(Debug)]
struct Charsmap {
    /// The double array, one unit per `u32`, exactly as the blob lays it out.
    trie: Box<[u32]>,
    /// The replacement strings, NUL separated. A unit's value is the byte offset of one.
    normalized: Box<str>,
    /// Bit `b` is set when some rule starts with byte `b`. A string holding none of those bytes
    /// cannot match anything and comes back borrowed.
    first: [u64; 4],
    /// The replacement for each ASCII byte, when it has one. Prose is ASCII, and this is the
    /// whole of what the charsmap does to it. Boxed to keep the normalizer enum small.
    ascii: Box<[Option<Box<str>>]>,
    /// CR LF is one grapheme, so it is looked up whole before its two bytes are looked up apart.
    crlf: Option<Box<str>>,
    /// Per code point, whether any rule can start at it: 0 unknown, 1 no, 2 yes. Asked once per
    /// character of non-ASCII input, so it answers from the map instead of walking the trie.
    starts: Box<[AtomicU32]>,
}

impl Clone for Charsmap {
    fn clone(&self) -> Self {
        Self {
            trie: self.trie.clone(),
            normalized: self.normalized.clone(),
            first: self.first,
            ascii: self.ascii.clone(),
            crlf: self.crlf.clone(),
            starts: self
                .starts
                .iter()
                .map(|cell| AtomicU32::new(cell.load(Ordering::Relaxed)))
                .collect(),
        }
    }
}

impl Charsmap {
    fn parse(charsmap: &[u8]) -> Result<Self> {
        if charsmap.len() < 4 {
            return Err("precompiled_charsmap is shorter than its length field".into());
        }
        let trie_bytes = u32::from_le_bytes(charsmap[..4].try_into().unwrap()) as usize;
        let blob_at = 4 + trie_bytes;
        if charsmap.len() < blob_at || !trie_bytes.is_multiple_of(4) {
            return Err("precompiled_charsmap is shorter than its trie".into());
        }
        // Safe to split: `trie_bytes` is a multiple of four, checked above.
        let (units, _) = charsmap[4..blob_at].as_chunks::<4>();
        let trie: Box<[u32]> = units.iter().map(|unit| u32::from_le_bytes(*unit)).collect();
        let normalized: Box<str> = std::str::from_utf8(&charsmap[blob_at..])
            .map_err(|_| -> crate::Error { "precompiled_charsmap does not end in UTF-8".into() })?
            .into();
        let mut map = Self {
            trie,
            normalized,
            first: [0; 4],
            ascii: vec![None; 128].into_boxed_slice(),
            crlf: None,
            starts: (0..0x1_0000).map(|_| AtomicU32::new(0)).collect(),
        };
        // The root unit's offset: `offset ^ byte` is the unit a rule's first byte lands on. A map
        // with no trie at all is a map with no rules, which a written config can hold.
        let Some(&root) = map.trie.first() else {
            return Ok(map);
        };
        let root = Unit(root).offset();
        for byte in 0..256u32 {
            let Some(&unit) = map.trie.get(root ^ byte as usize) else {
                continue;
            };
            if Unit(unit).label() == byte as usize {
                map.first[(byte / 64) as usize] |= 1 << (byte % 64);
            }
        }
        for byte in 0..128usize {
            map.ascii[byte] = map.lookup(&[byte as u8]).map(Into::into);
        }
        map.crlf = map.lookup(b"\r\n").map(Into::into);
        Ok(map)
    }

    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        // Text without a byte that starts a rule cannot be rewritten at all.
        if !self.may_match(input) {
            return Cow::Borrowed(input);
        }
        let bytes = input.as_bytes();
        let mut out: Option<String> = None;
        // `input[..copied]` is already in `out`; everything since is still to be copied.
        let mut copied = 0;
        let mut at = 0;
        while at < bytes.len() {
            // An ASCII byte can be normalized from the table, and taken in a run, as long as the
            // cut after it falls between two ASCII characters that are not CR LF: those are the
            // only cuts an ASCII grapheme does not cross.
            let mut end = at;
            while end < bytes.len()
                && bytes[end].is_ascii()
                && (end + 1 == bytes.len()
                    || (bytes[end + 1].is_ascii()
                        && !(bytes[end] == b'\r' && bytes[end + 1] == b'\n')))
            {
                end += 1;
            }
            if end > at {
                self.ascii_run(input, at, end, &mut out, &mut copied);
                at = end;
                continue;
            }
            // Everything else is graphemes, up to the next cut that is safe to make.
            let mut end = at + 1;
            while end < bytes.len()
                && !(bytes[end - 1].is_ascii()
                    && bytes[end].is_ascii()
                    && !(bytes[end - 1] == b'\r' && bytes[end] == b'\n'))
            {
                end += 1;
            }
            self.unicode_run(input, at, end, &mut out, &mut copied);
            at = end;
        }
        match out {
            Some(mut out) => {
                out.push_str(&input[copied..]);
                Cow::Owned(out)
            }
            None => Cow::Borrowed(input),
        }
    }

    /// One rewrite, and the run of input before it, into the output buffer.
    fn emit(
        &self,
        input: &str,
        at: usize,
        replacement: &str,
        out: &mut Option<String>,
        copied: &mut usize,
    ) {
        let out = out.get_or_insert_with(|| String::with_capacity(input.len()));
        out.push_str(&input[*copied..at]);
        out.push_str(replacement);
    }

    /// ASCII runs in the table, one byte (or one CR LF) at a time.
    fn ascii_run(
        &self,
        input: &str,
        from: usize,
        to: usize,
        out: &mut Option<String>,
        copied: &mut usize,
    ) {
        let bytes = &input.as_bytes()[from..to];
        let mut at = 0;
        while at < bytes.len() {
            let crlf = bytes[at] == b'\r' && bytes.get(at + 1) == Some(&b'\n');
            let (width, replacement) = match crlf {
                // CR LF is one grapheme, so it is looked up whole before its bytes are apart.
                true => match self.crlf.as_deref() {
                    Some(replacement) => (2, Some(replacement)),
                    None => (1, self.ascii[b'\r' as usize].as_deref()),
                },
                false => (1, self.ascii[bytes[at] as usize].as_deref()),
            };
            if let Some(replacement) = replacement {
                self.emit(input, from + at, replacement, out, copied);
                *copied = from + at + width;
            }
            at += width;
        }
    }

    /// Graphemes first, then the characters of every grapheme that did not match whole, which is
    /// what SentencePiece's normalizer does.
    ///
    /// Only the graphemes that hold a character a rule can start at are segmented: everything
    /// else cannot match, so the scan walks it a character at a time and leaves it alone.
    fn unicode_run(
        &self,
        input: &str,
        from: usize,
        to: usize,
        out: &mut Option<String>,
        copied: &mut usize,
    ) {
        let run = &input[from..to];
        let mut at = 0;
        while at < run.len() {
            let ch = run[at..].chars().next().unwrap();
            let next = input[from + at + ch.len_utf8()..].chars().next();
            if !self.starts_with_rule(ch, next) {
                at += ch.len_utf8();
                continue;
            }
            // The cluster that holds this character starts at the nearest boundary behind it.
            let mut start = at;
            while start > 0 {
                let mut cursor = GraphemeCursor::new(start, run.len(), true);
                if cursor.is_boundary(run, 0).unwrap_or(true) {
                    break;
                }
                start -= run[..start].chars().next_back().unwrap().len_utf8();
            }
            let end = {
                let mut cursor = GraphemeCursor::new(start, run.len(), true);
                cursor
                    .next_boundary(run, 0)
                    .ok()
                    .flatten()
                    .unwrap_or(run.len())
            };
            let grapheme = &run[start..end];
            let at_input = from + start;
            if grapheme.len() < 6
                && let Some(replacement) = self.lookup(grapheme.as_bytes())
            {
                self.emit(input, at_input, replacement, out, copied);
                *copied = at_input + grapheme.len();
            } else if grapheme.len() < 6
                && grapheme.len() == grapheme.chars().next().map_or(0, char::len_utf8)
            {
                // One character that did not match whole: the per-character pass would repeat
                // the lookup that just failed.
            } else {
                for (c_idx, character) in grapheme.char_indices() {
                    let part = &grapheme[c_idx..c_idx + character.len_utf8()];
                    if let Some(replacement) = self.lookup(part.as_bytes()) {
                        self.emit(input, at_input + c_idx, replacement, out, copied);
                        *copied = at_input + c_idx + part.len();
                    }
                }
            }
            at = end;
        }
    }

    /// Whether a lookup starting at `ch` can match, given the character that follows it.
    ///
    /// A rule key is one character or a character plus marks, so a character with no rule of its
    /// own still starts one when the trie continues into the next character.
    fn starts_with_rule(&self, ch: char, next: Option<char>) -> bool {
        let Some(cell) = self.starts.get(ch as usize) else {
            // Outside the BMP, and those are rare.
            return self.ends_at(ch).is_some_and(|(node, leaf)| {
                leaf || next.is_some_and(|next| self.continues(node, next))
            });
        };
        let (node, leaf) = match cell.load(Ordering::Relaxed) {
            0 => match self.ends_at(ch) {
                Some((node, leaf)) => {
                    cell.store(
                        (((node as u32) + 1) << 2) | ((leaf as u32) << 1),
                        Ordering::Relaxed,
                    );
                    (node, leaf)
                }
                None => {
                    cell.store(1, Ordering::Relaxed);
                    return false;
                }
            },
            1 => return false,
            cached => (((cached >> 2) - 1) as usize, cached & 2 != 0),
        };
        if leaf {
            return true;
        }
        next.is_some_and(|next| self.continues(node, next))
    }

    /// Where the walk ends after `ch`: the base its children XOR from, and whether the character
    /// is a rule of its own.
    fn ends_at(&self, ch: char) -> Option<(usize, bool)> {
        let mut buf = [0u8; 4];
        let bytes = ch.encode_utf8(&mut buf).as_bytes();
        let mut node = 0;
        let mut unit = Unit(*self.trie.first()?);
        node ^= unit.offset();
        for &byte in bytes {
            node ^= byte as usize;
            unit = Unit(*self.trie.get(node)?);
            if unit.label() != byte as usize {
                return None;
            }
            node ^= unit.offset();
        }
        Some((node, unit.has_leaf()))
    }

    /// Whether the trie continues from `node` into `next`, which is what makes a two-character
    /// key possible. A cell that is not a child fails the label check, so this cannot over-match.
    fn continues(&self, node: usize, next: char) -> bool {
        let mut buf = [0u8; 4];
        let byte = next.encode_utf8(&mut buf).as_bytes()[0];
        self.trie
            .get(node ^ byte as usize)
            .is_some_and(|unit| Unit(*unit).label() == byte as usize)
    }

    /// A string can only be rewritten when one of its bytes starts a rule.
    fn may_match(&self, input: &str) -> bool {
        input
            .as_bytes()
            .iter()
            .any(|&byte| self.first[(byte / 64) as usize] & (1 << (byte % 64)) != 0)
    }

    /// The value of the first rule that matches a prefix of `key`, which is the one
    /// SentencePiece applies, turned into its replacement string.
    fn lookup(&self, key: &[u8]) -> Option<&str> {
        let mut node = 0;
        let mut unit = Unit(*self.trie.first()?);
        node ^= unit.offset();
        for &byte in key {
            if byte == 0 {
                break;
            }
            node ^= byte as usize;
            unit = Unit(*self.trie.get(node)?);
            if unit.label() != byte as usize {
                return None;
            }
            node ^= unit.offset();
            if unit.has_leaf() {
                return self.text(Unit(*self.trie.get(node)?).value());
            }
        }
        None
    }

    fn text(&self, at: usize) -> Option<&str> {
        let bytes = self.normalized.as_bytes();
        let end = bytes.get(at..)?.iter().position(|&byte| byte == 0)? + at;
        self.normalized.get(at..end)
    }
}

/// One unit of the double array, with the bit layout SentencePiece puts it in.
#[derive(Clone, Copy)]
struct Unit(u32);

impl Unit {
    #[inline]
    fn has_leaf(self) -> bool {
        (self.0 >> 8) & 1 == 1
    }

    #[inline]
    fn value(self) -> usize {
        (self.0 & ((1 << 31) - 1)) as usize
    }

    #[inline]
    fn label(self) -> usize {
        (self.0 & ((1 << 31) | 0xFF)) as usize
    }

    #[inline]
    fn offset(self) -> usize {
        ((self.0 >> 10) as usize) << ((self.0 & (1 << 9)) >> 6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Normalize;

    /// A charsmap with two rules, `a` → `b` and `bc` → `d`, in the double-array layout
    /// SentencePiece writes: unit 0 is the root, and the unit at `base ^ label` carries the label,
    /// the leaf bit and an offset whose XOR lands on the next base, or on the value of a leaf.
    fn one_rule_charsmap() -> Vec<u8> {
        let mut trie = vec![0u32; 256];
        // `a` is a leaf, and its value is at 97 ^ 96 = 1.
        trie[b'a' as usize] = b'a' as u32 | 1 << 8 | 96 << 10;
        trie[1] = 0;
        // `b` is not a leaf: its base is 98 ^ 97 = 3, and `c` sits at 3 ^ 99 = 96.
        trie[b'b' as usize] = b'b' as u32 | 97 << 10;
        trie[96] = b'c' as u32 | 1 << 8 | 100 << 10;
        // `c`'s value is at 96 ^ 100 = 4, offset 2 into the blob.
        trie[4] = 2;
        let mut bytes = (trie.len() as u32 * 4).to_le_bytes().to_vec();
        bytes.extend(trie.iter().flat_map(|unit| unit.to_le_bytes()));
        bytes.extend(b"b\0d\0");
        bytes
    }

    fn normalize(input: &str) -> String {
        let normalizer = PrecompiledNormalizer::from_charsmap(&one_rule_charsmap()).unwrap();
        match normalizer.normalize(input, 0).unwrap() {
            Cow::Borrowed(input) => input.to_string(),
            Cow::Owned(out) => out,
        }
    }

    #[test]
    fn a_rule_rewrites_and_everything_else_is_borrowed() {
        assert_eq!(normalize("a"), "b");
        assert_eq!(normalize("xax"), "xbx");
        assert_eq!(normalize("aa"), "bb");
        assert_eq!(normalize("xy"), "xy");
        assert_eq!(normalize(""), "");
        // CR LF is one grapheme, but with no rule of its own its two bytes are looked up apart.
        assert_eq!(normalize("a\r\n"), "b\r\n");
        // Non-ASCII input takes the grapheme path and agrees.
        assert_eq!(normalize("aéa"), "béb");
    }

    /// The scan skips a character only when no rule can start at it: not a rule of its own, and
    /// no rule continues into the character after it. `b` is only the first half of `bc`.
    #[test]
    fn a_character_starts_a_rule_only_when_the_map_can_use_it() {
        let normalizer = PrecompiledNormalizer::from_charsmap(&one_rule_charsmap()).unwrap();
        let map = &normalizer.map;
        assert!(map.starts_with_rule('a', None));
        assert!(map.starts_with_rule('a', Some('b')));
        assert!(!map.starts_with_rule('b', None));
        assert!(map.starts_with_rule('b', Some('c')));
        assert!(!map.starts_with_rule('b', Some('a')));
        assert!(!map.starts_with_rule('z', Some('a')));
    }

    #[test]
    fn a_charsmap_with_no_trie_has_no_rules() {
        let normalizer = PrecompiledNormalizer::from_charsmap(&0u32.to_le_bytes()).unwrap();
        assert!(matches!(
            normalizer.normalize("a\r\né", 0).unwrap(),
            Cow::Borrowed("a\r\né")
        ));
    }
}
