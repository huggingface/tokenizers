use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
pub use spm_precompiled::Precompiled;
use unicode_segmentation::UnicodeSegmentation;

/// Double-array unit accessors. These mirror the `ArrayUnit` bit layout `spm_precompiled`
/// parses internally and keeps private, so this module can walk the same trie itself.
fn has_leaf(unit: usize) -> bool {
    (unit >> 8) & 1 == 1
}
fn leaf_value(unit: usize) -> usize {
    unit & ((1usize << 31) - 1)
}
fn label(unit: usize) -> usize {
    unit & ((1usize << 31) | 0xFF)
}
fn offset(unit: usize) -> usize {
    (unit >> 10) << ((unit & (1usize << 9)) >> 6)
}

/// The parsed `precompiled_charsmap`: the double-array trie plus the NUL-separated
/// replacement strings.
///
/// This mirrors the layout `spm_precompiled` parses internally (a `u32` little-endian trie
/// byte length, then that many bytes of `u32` little-endian trie units, then the
/// replacements blob). It exists because `spm_precompiled` keeps its parse private and only
/// exposes a *shortest*-prefix lookup, which is the wrong primitive for SentencePiece
/// normalization: at each position the *longest* matching rule must win.
#[derive(Debug, Clone)]
struct CharsmapMatcher {
    trie: Vec<usize>,
    normalized: String,
}

impl CharsmapMatcher {
    fn parse(charsmap: &[u8]) -> std::result::Result<Self, String> {
        if charsmap.len() < 4 {
            return Err("precompiled_charsmap is shorter than its 4-byte trie length".into());
        }
        let trie_len =
            u32::from_le_bytes(charsmap[0..4].try_into().expect("length checked above")) as usize;
        if !trie_len.is_multiple_of(4) || charsmap.len() < 4 + trie_len {
            return Err(format!(
                "precompiled_charsmap trie length {trie_len} does not fit its {} bytes",
                charsmap.len()
            ));
        }
        let (chunks, remainder) = charsmap[4..4 + trie_len].as_chunks::<4>();
        debug_assert!(remainder.is_empty(), "trie length checked above");
        let trie = chunks
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk) as usize)
            .collect();
        let normalized = String::from_utf8(charsmap[4 + trie_len..].to_vec())
            .map_err(|_| "precompiled_charsmap replacements are not valid UTF-8".to_string())?;
        Ok(Self { trie, normalized })
    }

    /// Longest rule key that prefixes `input`, as `(key byte length, replacement)`.
    ///
    /// The byte length is what the caller must consume: keys are valid UTF-8, so consuming
    /// exactly the key keeps the cursor on a char boundary.
    fn longest_match(&self, input: &[u8]) -> Option<(usize, &str)> {
        let mut node_pos = 0usize;
        let mut unit = *self.trie.first()?;
        node_pos ^= offset(unit);
        // `best` is the longest key seen so far: later (longer) matches overwrite it.
        let mut best: Option<(usize, usize)> = None;
        let mut len = 0usize;
        for &byte in input {
            if byte == 0 {
                break;
            }
            node_pos ^= byte as usize;
            unit = *self.trie.get(node_pos)?;
            if label(unit) != byte as usize {
                break;
            }
            len += 1;
            node_pos ^= offset(unit);
            if has_leaf(unit) {
                let value = *self.trie.get(node_pos)?;
                best = Some((len, leaf_value(value)));
            }
        }
        let (len, value) = best?;
        // `value` is a byte offset into the NUL-separated replacements; the replacement
        // runs to the next NUL (or the end of the blob).
        let replacement = self.normalized.get(value..)?;
        let end = replacement.find('\0').unwrap_or(replacement.len());
        Some((len, &replacement[..end]))
    }
}

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
/// The bytes are also parsed once, up front, into a [`CharsmapMatcher`] so that normalization
/// can do SentencePiece-compatible longest-prefix matching instead of `spm_precompiled`'s
/// shortest-prefix lookup.
///
/// [`charsmap`](Self::charsmap) is an `Option` because not every producer has the bytes to hand:
/// one that only has an already-parsed [`Precompiled`] records `None`, and a writer then reports
/// that rather than inventing a blob. `from_charsmap` is currently the only constructor, so today
/// it is always `Some`.
#[derive(Debug, Clone)]
pub struct PrecompiledNormalizer {
    parsed: Precompiled,
    charsmap: Option<Box<[u8]>>,
    matcher: CharsmapMatcher,
}

impl PrecompiledNormalizer {
    /// Parse a charsmap and keep it, which is what a reader does.
    pub fn from_charsmap(charsmap: &[u8]) -> Result<Self> {
        let parsed =
            Precompiled::from(charsmap).map_err(|e| -> crate::Error { e.to_string().into() })?;
        let matcher = CharsmapMatcher::parse(charsmap).map_err(|e| -> crate::Error { e.into() })?;
        Ok(Self {
            parsed,
            charsmap: Some(charsmap.into()),
            matcher,
        })
    }

    /// The bytes this was parsed from, when they are known.
    pub fn charsmap(&self) -> Option<&[u8]> {
        self.charsmap.as_deref()
    }

    /// The parsed value, for anything that wants to normalize with it directly.
    pub fn parsed(&self) -> &Precompiled {
        &self.parsed
    }
}

impl pipeline::Normalizer for PrecompiledNormalizer {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        // SentencePiece-compatible normalization: leftmost-longest prefix matching over the
        // byte stream. At each position the longest rule whose key prefixes the remaining
        // input wins; its replacement is emitted and exactly the key's bytes are consumed.
        // Positions covered by no rule are copied through unchanged.
        //
        // This deliberately does not reuse `spm_precompiled`'s matching, which only exposes a
        // shortest-prefix lookup. The previous grapheme walk built on it (a) preferred the
        // *shortest* rule when a longer one matched, (b) replaced the whole grapheme with the
        // short match and silently dropped the bytes the match did not consume, and (c) skipped
        // whole-grapheme lookup for graphemes of 6+ bytes entirely, shredding them per
        // codepoint instead.
        let bytes = input.as_bytes();
        let mut transformed: Option<String> = None;
        let mut pos = 0;
        while pos < bytes.len() {
            if let Some((consumed, replacement)) = self.matcher.longest_match(&bytes[pos..]) {
                let string = transformed.get_or_insert_with(|| {
                    let mut s = String::with_capacity(input.len());
                    s.push_str(&input[..pos]);
                    s
                });
                string.push_str(replacement);
                // `consumed` is the byte length of a rule key that prefixes `input[pos..]`;
                // keys are valid UTF-8, so `pos` stays on a char boundary. It is also
                // non-zero: a match is only reported after consuming at least one byte.
                pos += consumed;
            } else if let Some(character) = input[pos..].chars().next() {
                if let Some(string) = transformed.as_mut() {
                    string.push(character);
                }
                pos += character.len_utf8();
            } else {
                // Unreachable: `pos < bytes.len()` guarantees a next char. Break rather
                // than spin forever if that invariant ever fails.
                break;
            }
        }
        if let Some(string) = transformed {
            Ok(Cow::Owned(string))
        } else {
            Ok(Cow::Borrowed(input))
        }
    }
}

impl pipeline::Normalizer for Precompiled {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        // NOTE: this keeps the historical grapheme-walk / shortest-prefix behavior for direct
        // users of `Precompiled`. `PrecompiledNormalizer` (what the pipeline runs) no longer
        // delegates here: it normalizes with its own longest-prefix matcher instead. This impl
        // cannot do longest-prefix matching because the trie `Precompiled` was built from is
        // not visible through its API.
        let mut transformed: Option<String> = None;
        for (g_idx, grapheme) in input.grapheme_indices(true) {
            if grapheme.len() < 6
                && let Some(replacement) = self.transform(grapheme)
            {
                let string = transformed.get_or_insert_with(|| {
                    let mut s = String::with_capacity(input.len());
                    s.push_str(&input[..g_idx]);
                    s
                });
                string.push_str(replacement);
                continue;
            }
            for (c_idx, character) in grapheme.char_indices() {
                if let Some(replacement) =
                    self.transform(&grapheme[c_idx..c_idx + character.len_utf8()])
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
        if let Some(string) = transformed {
            Ok(Cow::Owned(string))
        } else {
            Ok(Cow::Borrowed(input))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Normalizer;

    /// Build a minimal `precompiled_charsmap` blob for `rules: &[(`key`, `replacement`)]`.
    ///
    /// Lays out a simple (non-compact) double-array trie in exactly the format
    /// [`CharsmapMatcher::parse`] reads: a `u32` little-endian trie byte length, then that
    /// many bytes of `u32` little-endian trie units, then the NUL-separated replacements.
    /// Unit encoding mirrors `spm_precompiled`'s `ArrayUnit` bit layout.
    fn build_charsmap(rules: &[(&str, &str)]) -> Vec<u8> {
        // Replacement strings, NUL-separated; remember each rule's byte offset.
        let mut normalized = String::new();
        let mut rule_values = Vec::with_capacity(rules.len());
        for (_, replacement) in rules {
            rule_values.push(normalized.len());
            normalized.push_str(replacement);
            normalized.push('\0');
        }

        // Byte-trie of the rule keys.
        #[derive(Default)]
        struct Node {
            children: Vec<(u8, usize)>,
            value: Option<usize>,
        }
        let mut nodes = vec![Node::default()];
        for (rule, (key, _)) in rules.iter().enumerate() {
            let mut node = 0;
            for &byte in key.as_bytes() {
                assert!(byte != 0, "test rule keys must not contain NUL");
                node = match nodes[node].children.iter().find(|(b, _)| *b == byte) {
                    Some((_, child)) => *child,
                    None => {
                        nodes.push(Node::default());
                        let child = nodes.len() - 1;
                        nodes[node].children.push((byte, child));
                        child
                    }
                };
            }
            nodes[node].value = Some(rule_values[rule]);
        }

        // Give every node a fixed position `p = (id + 1) * 512`. A node's child for byte `b`
        // lives at `p ^ b`, which stays inside `[p, p + 255]` and therefore can never collide
        // with another node's slots; a leaf child's replacement offset lives at the child's
        // own position, which no child slot can equal for non-zero `b`.
        let pos = |id: usize| (id + 1) * 512;
        let mut trie = vec![0usize; pos(nodes.len()) + 256];
        trie[0] = pos(0) << 10; // root unit: offset with bit 9 clear
        for (id, node) in nodes.iter().enumerate() {
            for &(byte, child) in &node.children {
                let slot = pos(id) ^ byte as usize;
                // Offset the walk must apply after matching this child to land on its position.
                let child_offset = slot ^ pos(child);
                let leaf = nodes[child].value.is_some();
                trie[slot] = (child_offset << 10) | ((leaf as usize) << 8) | byte as usize;
                if let Some(value) = nodes[child].value {
                    trie[pos(child)] = value;
                }
            }
        }

        let mut blob = ((trie.len() * 4) as u32).to_le_bytes().to_vec();
        for unit in &trie {
            blob.extend_from_slice(&(*unit as u32).to_le_bytes());
        }
        blob.extend_from_slice(normalized.as_bytes());
        blob
    }

    fn normalizer(rules: &[(&str, &str)]) -> PrecompiledNormalizer {
        PrecompiledNormalizer::from_charsmap(&build_charsmap(rules)).expect("test charsmap parses")
    }

    #[test]
    fn longest_rule_wins_over_shorter_prefix() {
        // "a" is a proper prefix of "a\u{301}". The old shortest-prefix lookup normalized
        // the whole grapheme to "X"; SentencePiece takes the longest match, "Y".
        let n = normalizer(&[("a", "X"), ("a\u{301}", "Y")]);
        assert_eq!(n.normalize("a\u{301}", 0).unwrap().into_owned(), "Y");
    }

    #[test]
    fn unmatched_tail_is_copied_through() {
        let n = normalizer(&[("a", "X"), ("a\u{301}", "Y")]);
        assert_eq!(n.normalize("a\u{301}!", 0).unwrap().into_owned(), "Y!");
        assert_eq!(n.normalize("ab", 0).unwrap().into_owned(), "Xb");
        assert_eq!(n.normalize("q", 0).unwrap().into_owned(), "q");
    }

    #[test]
    fn long_grapheme_rules_apply() {
        // A 7-byte single grapheme: the old walk skipped whole-grapheme lookup for
        // graphemes of 6+ bytes and shredded it per codepoint ("X\u{301}\u{302}\u{303}").
        let n = normalizer(&[("a", "X"), ("a\u{301}\u{302}\u{303}", "W")]);
        assert_eq!(
            n.normalize("a\u{301}\u{302}\u{303}", 0)
                .unwrap()
                .into_owned(),
            "W"
        );
    }

    #[test]
    fn untouched_input_is_borrowed() {
        let n = normalizer(&[("a", "X")]);
        assert!(matches!(n.normalize("hello", 0).unwrap(), Cow::Borrowed(_)));
    }
}
