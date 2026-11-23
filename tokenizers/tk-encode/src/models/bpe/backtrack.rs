//! Backtracking BPE: the encoding of vocabularies whose token ids are their ranks. The algorithm is
//! the one in [github/rust-gems `bpe`](https://github.com/github/rust-gems/tree/main/crates/bpe)
//! (MIT).
use ahash::AHashMap;

use crate::pipeline::Token;
use crate::tokenizer::Result;

const NONE: u32 = u32::MAX;

pub(crate) struct Backtracking {
    /// Each token's length in bytes, by id.
    lens: Vec<u32>,
    ids: AHashMap<Box<[u8]>, u32>,
    max_token_len: usize,
    /// The longest token that is a proper prefix of each token, or `NONE`.
    next_prefix: Vec<u32>,
    /// The two tokens each token merges from. A token with no valid split points back to itself.
    split: Vec<(u32, u32)>,
    /// `(left, right) -> merged` for every split in `split`.
    merged: AHashMap<(u32, u32), u32>,
}

impl Backtracking {
    /// Build from the vocabulary's tokens, indexed by id.
    pub(crate) fn new(tokens: &[&[u8]]) -> Self {
        let lens = tokens.iter().map(|t| t.len() as u32).collect();
        let ids = tokens
            .iter()
            .zip(0..)
            .map(|(token, id)| (Box::<[u8]>::from(*token), id))
            .collect();
        let max_token_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        let mut engine = Self {
            lens,
            ids,
            max_token_len,
            next_prefix: Vec::new(),
            split: Vec::with_capacity(tokens.len()),
            merged: AHashMap::new(),
        };
        engine.next_prefix = tokens
            .iter()
            .map(|token| match token.split_last() {
                Some((_, head)) if !head.is_empty() => engine.longest_prefix(head),
                _ => NONE,
            })
            .collect();

        for (id, token) in tokens.iter().enumerate() {
            let id = id as u32;
            let mut left = engine.next_prefix[id as usize];
            let mut split = (id, id);
            while left != NONE {
                let rest = &token[engine.len(left)..];
                if let Some(&right) = engine.ids.get(rest)
                    && left < id
                    && right < id
                    && engine.is_valid_pair(left, right)
                {
                    engine.merged.insert((left, right), id);
                    split = (left, right);
                    break;
                }
                left = engine.next_prefix[left as usize];
            }
            engine.split.push(split);
        }
        engine
    }

    #[inline]
    fn len(&self, id: u32) -> usize {
        self.lens[id as usize] as usize
    }

    /// The longest token that `text` starts with, or `NONE`.
    fn longest_prefix(&self, text: &[u8]) -> u32 {
        (1..=text.len().min(self.max_token_len))
            .rev()
            .find_map(|len| self.ids.get(&text[..len]).copied())
            .unwrap_or(NONE)
    }

    /// Whether BPE can emit `right` right after `left`: walk the boundary between them inward
    /// through both merge trees, and fail on any merge across it that ranks before the tokens on
    /// either side were formed.
    fn is_valid_pair(&self, mut left: u32, mut right: u32) -> bool {
        let mut limit = u32::MAX;
        loop {
            if let Some(&merged) = self.merged.get(&(left, right))
                && merged < limit
            {
                return false;
            }
            if left > right {
                limit = left;
                left = self.split[left as usize].1;
                if left == limit {
                    limit = right + 1;
                    right = self.split[right as usize].0;
                    if right + 1 == limit {
                        return true;
                    }
                }
            } else {
                limit = right + 1;
                right = self.split[right as usize].0;
                if right + 1 == limit {
                    limit = left;
                    left = self.split[left as usize].1;
                    if left == limit {
                        return true;
                    }
                }
            }
        }
    }

    /// Encode `text`, appending its tokens to `out`. `dead_ends` is scratch: one bit per position
    /// that no token may end at.
    pub(crate) fn encode(
        &self,
        text: &[u8],
        out: &mut Vec<Token>,
        dead_ends: &mut Vec<u32>,
    ) -> Result<()> {
        dead_ends.clear();
        dead_ends.resize(text.len() / 32 + 1, 0);
        let is_dead = |dead_ends: &[u32], at: usize| dead_ends[at / 32] & (1 << (at % 32)) != 0;

        let first = out.len();
        let mut pos = 0;
        let mut candidate = self.longest_prefix(text);
        while pos < text.len() {
            let previous = (out.len() > first).then(|| out[out.len() - 1].id());
            while candidate != NONE {
                let end = pos + self.len(candidate);
                if !is_dead(dead_ends, end)
                    && previous.is_none_or(|previous| self.is_valid_pair(previous, candidate))
                {
                    break;
                }
                candidate = self.next_prefix[candidate as usize];
            }
            if candidate != NONE {
                out.push(Token::from(candidate));
                pos += self.len(candidate);
                candidate = self.longest_prefix(&text[pos..]);
                continue;
            }
            dead_ends[pos / 32] |= 1 << (pos % 32);
            if out.len() == first {
                return Err(format!(
                    "backtracking BPE found no encoding for {:?}",
                    String::from_utf8_lossy(text)
                )
                .into());
            }
            let popped = out.pop().map_or(NONE, Token::id);
            pos -= self.len(popped);
            candidate = self.next_prefix[popped as usize];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_like_ranked_merging() {
        let cases: &[(&[&str], &str, &[u32])] = &[
            (&["a", "b", "c", "ab", "ba", "aba"], "aba", &[5]),
            (&["a", "b", "c", "ab", "ba", "aba"], "abab", &[3, 3]),
            (&["a", "b", "c", "ab", "ba", "aba"], "cab", &[2, 3]),
            (&["a", "b", "c", "bc", "ab"], "abc", &[0, 3]),
        ];
        for &(tokens, text, want) in cases {
            let tokens: Vec<&[u8]> = tokens.iter().map(|t| t.as_bytes()).collect();
            let mut out = Vec::new();
            Backtracking::new(&tokens)
                .encode(text.as_bytes(), &mut out, &mut Vec::new())
                .unwrap();
            let got: Vec<u32> = out.into_iter().map(Token::id).collect();
            assert_eq!(got, want, "{text:?} over {tokens:?}");
        }
    }
}
