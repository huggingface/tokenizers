//! Tokens the merge list cannot build, emitted whole when they start a pretoken.
//!
//! Some tiktoken vocabularies hold tokens no pair of earlier tokens can build (Inkling-2 has
//! 60). They are unreachable by the merges, so they only appear when a whole pretoken happens
//! to equal one. Rather than teach the encoder about malformed vocabularies, this wraps it:
//! a pretoken starting with one of those tokens emits it and the inner model encodes the rest.
//!
//! Nothing here is consulted unless [`BpeConfig::prefix_atoms`](super::BpeConfig) asked for it.

use ahash::AHashSet;

use crate::models::bpe::{BPE, BpeConfig, BpeScratch, Merges, Vocab};
use crate::pipeline::{Token, Tokenize};
use crate::tokenizer::Result;
use crate::utils::byte_level;

pub struct PrefixAtoms {
    inner: BPE,
    /// Longest first, so the first match is the longest.
    roots: Vec<(Box<[u8]>, u32)>,
}

impl PrefixAtoms {
    pub fn from_config(config: BpeConfig) -> Result<Self> {
        let roots = unreachable_roots(&config.vocab, &config.merges, config.byte_level);
        let inner = BPE::from_config(config)?;
        Ok(Self { inner, roots })
    }

    pub fn inner(&self) -> &BPE {
        &self.inner
    }

    /// Emit every root the sequence starts with and return what is left.
    fn emit_roots<'a>(&self, mut sequence: &'a str, output: &mut Vec<Token>) -> &'a str {
        while let Some((rest, id)) = self.roots.iter().find_map(|(token, id)| {
            let rest = sequence.as_bytes().strip_prefix(&**token)?;
            std::str::from_utf8(rest).ok().map(|rest| (rest, *id))
        }) {
            output.push(Token::from(id));
            sequence = rest;
            if sequence.is_empty() {
                break;
            }
        }
        sequence
    }
}

impl Tokenize for PrefixAtoms {
    type Scratch = BpeScratch;

    fn tokenize(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<Token>,
    ) -> Result<()> {
        let rest = self.emit_roots(sequence, output);
        if rest.len() == sequence.len() {
            return self.inner.tokenize(sequence, scratch, output);
        }
        if !rest.is_empty() {
            self.inner.tokenize(rest, scratch, output)?;
        }
        Ok(())
    }

    fn init_scratch(&self) -> Self::Scratch {
        self.inner.init_scratch()
    }
}

/// Every vocabulary token no merge list can build, longest first.
fn unreachable_roots(vocab: &Vocab, merges: &Merges, byte_level: bool) -> Vec<(Box<[u8]>, u32)> {
    let pairs: AHashSet<(&str, &str)> = merges
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let mut tokens: Vec<(&str, u32)> = vocab.iter().map(|(s, &id)| (s.as_str(), id)).collect();
    tokens.sort_by_key(|(token, _)| token.len());
    let mut seen: AHashSet<u32> = AHashSet::new();
    let mut roots = Vec::new();
    for (token, id) in tokens {
        let ok = token.chars().count() == 1
            || token.char_indices().skip(1).any(|(i, _)| {
                let (a, b) = token.split_at(i);
                matches!((vocab.get(a), vocab.get(b)), (Some(&x), Some(&y))
                    if seen.contains(&x) && seen.contains(&y) && pairs.contains(&(a, b)))
            });
        if ok {
            seen.insert(id);
        } else if token.chars().count() > 1 {
            let bytes: Box<[u8]> = if byte_level {
                token.chars().flat_map(byte_level::reverse_lookup).collect()
            } else {
                token.as_bytes().into()
            };
            roots.push((bytes, id));
        }
    }
    roots.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
    roots
}
