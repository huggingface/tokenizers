use crate::tokenizer::{Offsets, Token};
use crate::utils::padding::PaddingDirection;
use rayon::prelude::*;

/// Represents the output of a `Tokenizer`.
#[derive(Default, PartialEq, Debug, Clone)]
pub struct Encoding {
    /// IDs produced by the `Tokenizer`
    ids: Vec<u32>,
    /// Type of the IDs
    type_ids: Vec<u32>,
    /// Tokens associated to each ID
    tokens: Vec<String>,
    /// Indice of the word associated to each token/ID
    words: Vec<u32>,
    /// Offsets of the token/ID from the NormalizedString
    offsets: Vec<Offsets>,
    /// Mask identifying special tokens
    special_tokens_mask: Vec<u32>,
    /// Mask identifying padding tokens for the attention mechanism
    attention_mask: Vec<u32>,
    /// A list of overflowing Encoding generated when we got truncated
    overflowing: Vec<Encoding>,
}
impl Encoding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        words: Vec<u32>,
        offsets: Vec<Offsets>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        overflowing: Vec<Encoding>,
    ) -> Self {
        Encoding {
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
        }
    }

    pub fn from_tokens(tokens: Vec<Token>, type_id: u32) -> Self {
        let length = tokens.len();
        let (ids, tokens, offsets, words) = tokens.into_iter().fold(
            (
                Vec::with_capacity(length),
                Vec::with_capacity(length),
                Vec::with_capacity(length),
                Vec::with_capacity(length),
            ),
            |(mut ids, mut tokens, mut offsets, mut words), t| {
                ids.push(t.id);
                tokens.push(t.value);
                offsets.push(t.offsets);
                words.push(t.word);
                (ids, tokens, offsets, words)
            },
        );

        Encoding {
            ids,
            tokens,
            offsets,
            words,
            type_ids: vec![type_id; length],
            attention_mask: vec![1; length],
            special_tokens_mask: vec![0; length],
            overflowing: vec![],
        }
    }

    pub fn get_tokens(&self) -> &[String] {
        &self.tokens[..]
    }

    pub fn get_words(&self) -> &[u32] {
        &self.words
    }

    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    pub fn get_offsets(&self) -> &[Offsets] {
        &self.offsets
    }

    pub fn get_offsets_mut(&mut self) -> &mut [Offsets] {
        &mut self.offsets
    }

    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn get_overflowing(&self) -> &Vec<Encoding> {
        &self.overflowing
    }

    pub fn take_overflowing(&mut self) -> Vec<Encoding> {
        std::mem::replace(&mut self.overflowing, vec![])
    }

    /// Find the boundaries of the word at the given index (Indices in the various Vec).
    /// The upper value is excluded, just like for offsets.
    pub fn word_boundaries(&self, index: usize) -> Option<(usize, usize)> {
        self.words.get(index).map(|word_id| {
            let start = (0..=index)
                .rev()
                .take_while(|i| self.words[*i] == *word_id)
                .min()
                .unwrap();

            let end = (index..self.words.len())
                .take_while(|i| self.words[*i] == *word_id)
                .max()
                .unwrap();

            (start, end + 1)
        })
    }

    /// Find the Offsets of the word that contains the character at the specified position
    pub fn char_to_word_offsets(&self, pos: usize) -> Option<Offsets> {
        self.offsets
            .iter()
            .enumerate()
            .find(|(_, (start, end))| pos >= *start && pos < *end)
            .map(|(index, _)| self.token_to_word_offsets(index))
            .flatten()
    }

    /// Find the Offsets of the token that contains the character at the specified position
    pub fn char_to_token_offsets(&self, pos: usize) -> Option<Offsets> {
        self.char_to_token(pos).map(|i| self.offsets[i])
    }

    /// Find the Offsets of the word that contains the token at the given index
    pub fn token_to_word_offsets(&self, index: usize) -> Option<Offsets> {
        self.word_boundaries(index)
            .map(|(min, max)| (self.offsets[min].0, self.offsets[max - 1].1))
    }

    /// Return the index of the token at position of the given char.
    pub fn char_to_token(&self, pos: usize) -> Option<usize> {
        self.offsets
            .iter()
            .position(|(start, end)| pos >= *start && pos < *end)
    }

    /// Truncate the current `Encoding`.
    ///
    /// Panic if `stride >= max_len` or `max_len == 0`.
    pub fn truncate(&mut self, max_len: usize, stride: usize) {
        if max_len >= self.ids.len() {
            return;
        }
        // We only truncate if max_len > 0, it makes no sense otherwise
        assert!(max_len > 0);

        // Get the main overflowing part
        let o_ids = self.ids.split_off(max_len);
        let o_type_ids = self.type_ids.split_off(max_len);
        let o_tokens = self.tokens.split_off(max_len);
        let o_words = self.words.split_off(max_len);
        let o_offsets = self.offsets.split_off(max_len);
        let o_spe_toks = self.special_tokens_mask.split_off(max_len);
        let o_attent = self.attention_mask.split_off(max_len);

        // Now we need to separate the overflowing part into as many Encoding as needed
        assert!(stride < max_len);
        let part_size = max_len - stride;
        let mut overflowing = vec![];
        let mut part_id = 0;
        let mut prev_encoding: &Encoding = self;

        loop {
            if part_size * part_id >= o_ids.len() {
                break;
            }

            let o = Encoding {
                ids: get_current_part(&prev_encoding.ids, &o_ids, part_size, part_id, stride),
                type_ids: get_current_part(
                    &prev_encoding.type_ids,
                    &o_type_ids,
                    part_size,
                    part_id,
                    stride,
                ),
                tokens: get_current_part(
                    &prev_encoding.tokens,
                    &o_tokens,
                    part_size,
                    part_id,
                    stride,
                ),
                words: get_current_part(&prev_encoding.words, &o_words, part_size, part_id, stride),
                offsets: get_current_part(
                    &prev_encoding.offsets,
                    &o_offsets,
                    part_size,
                    part_id,
                    stride,
                ),
                special_tokens_mask: get_current_part(
                    &prev_encoding.special_tokens_mask,
                    &o_spe_toks,
                    part_size,
                    part_id,
                    stride,
                ),
                attention_mask: get_current_part(
                    &prev_encoding.attention_mask,
                    &o_attent,
                    part_size,
                    part_id,
                    stride,
                ),
                overflowing: vec![],
            };

            part_id += 1;
            overflowing.push(o);
            prev_encoding = &overflowing.last().unwrap();
        }

        self.overflowing = overflowing;
    }

    /// Merge all Encodings together
    pub fn merge(encodings: &[Encoding], growing_offsets: bool) -> Encoding {
        if encodings.is_empty() {
            return Encoding::default();
        }

        let (firsts, others) = encodings.split_at(1);
        let mut first: Encoding = firsts[0].clone();

        for encoding in others {
            first.merge_with(encoding.clone(), growing_offsets);
        }

        first
    }

    /// Merge ourself with the given `Encoding`. Happens in place.
    pub fn merge_with(&mut self, pair: Encoding, growing_offsets: bool) {
        // Handle merging the overflowing parts too: Combine them all
        // In most of the cases, we expect `pair.overflowing.len() == 0`
        let mut overflowings = vec![];

        // 1. All our overflowings with all the others
        for self_o in &self.overflowing {
            // 1. The pair itself
            let mut n_encoding = self_o.clone();
            n_encoding.merge_with(pair.clone(), growing_offsets);
            overflowings.push(n_encoding);

            // 2. Its overflowings (this should rarely happen...)
            for other_o in &pair.overflowing {
                let mut n_encoding = self_o.clone();
                n_encoding.merge_with(other_o.clone(), growing_offsets);
                overflowings.push(n_encoding);
            }
        }
        // 2. Ourself with all the other overflowings (this should rarely happen too...)
        for other_o in &pair.overflowing {
            let mut n_encoding = self.clone();
            n_encoding.merge_with(other_o.clone(), growing_offsets);
            overflowings.push(n_encoding);
        }

        // Finish by merging ourself with the other encoding
        self.ids.extend(pair.ids);
        self.type_ids.extend(pair.type_ids);
        self.tokens.extend(pair.tokens);

        let starting_word = self.words.last().map_or(0, |w| *w + 1);
        self.words.extend(
            pair.words
                .into_iter()
                .map(|w| w + starting_word)
                .collect::<Vec<_>>(),
        );

        let starting_offset = if growing_offsets {
            self.offsets.last().map_or(0, |o| o.1)
        } else {
            0
        };
        self.offsets.extend(
            pair.offsets
                .into_iter()
                .map(|(start, end)| (start + starting_offset, end + starting_offset))
                .collect::<Vec<_>>(),
        );
        self.special_tokens_mask.extend(pair.special_tokens_mask);
        self.attention_mask.extend(pair.attention_mask);
        self.overflowing = overflowings;
    }

    pub fn pad(
        &mut self,
        target_length: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        direction: PaddingDirection,
    ) {
        // Dispatch call to all the overflowings first
        self.overflowing.par_iter_mut().for_each(|encoding| {
            encoding.pad(target_length, pad_id, pad_type_id, pad_token, direction)
        });

        // Then check if we should pad ourself
        if self.ids.len() >= target_length {
            // We just do nothing if the wanted padding length is smaller than us
            return;
        }
        let pad_length = target_length - self.ids.len();

        match direction {
            PaddingDirection::Left => {
                self.ids = (0..pad_length)
                    .map(|_| pad_id)
                    .chain(self.ids.drain(..))
                    .collect();
                self.type_ids = (0..pad_length)
                    .map(|_| pad_type_id)
                    .chain(self.type_ids.drain(..))
                    .collect();
                self.tokens = (0..pad_length)
                    .map(|_| pad_token.to_owned())
                    .chain(self.tokens.drain(..))
                    .collect();
                self.words = (0..pad_length)
                    .map(|_| 0)
                    .chain(self.words.drain(..).map(|w| w + 1))
                    .collect();
                self.attention_mask = (0..pad_length)
                    .map(|_| 0)
                    .chain(self.attention_mask.drain(..))
                    .collect();
                self.special_tokens_mask = (0..pad_length)
                    .map(|_| 1)
                    .chain(self.special_tokens_mask.drain(..))
                    .collect();
                self.offsets = (0..pad_length)
                    .map(|_| (0, 0))
                    .chain(self.offsets.drain(..))
                    .collect();
            }
            PaddingDirection::Right => {
                self.ids.extend((0..pad_length).map(|_| pad_id));
                self.type_ids.extend((0..pad_length).map(|_| pad_type_id));
                self.tokens
                    .extend((0..pad_length).map(|_| pad_token.to_owned()));
                let padding_word = self.words.last().map_or(0, |w| *w + 1);
                self.words.extend((0..pad_length).map(|_| padding_word));
                self.attention_mask.extend((0..pad_length).map(|_| 0));
                self.special_tokens_mask.extend((0..pad_length).map(|_| 1));
                self.offsets.extend((0..pad_length).map(|_| (0, 0)));
            }
        }
    }
}

#[inline]
fn get_current_part<T: Clone>(
    prev: &[T],
    current: &[T],
    size: usize,
    idx: usize,
    stride: usize,
) -> Vec<T> {
    let curr_slice = if (idx + 1) * size > current.len() {
        &current[idx * size..]
    } else {
        &current[idx * size..(idx + 1) * size]
    };
    let prev_slice = &prev[prev.len() - stride..];
    [prev_slice, curr_slice].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_encodings() {
        let mut a = Encoding {
            ids: vec![1],
            type_ids: vec![0],
            tokens: vec![String::from("Hello ")],
            words: vec![0],
            offsets: vec![(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            overflowing: vec![],
        };
        let b = Encoding {
            ids: vec![2],
            type_ids: vec![1],
            tokens: vec![String::from("World!")],
            words: vec![0],
            offsets: vec![(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            overflowing: vec![],
        };
        a.merge_with(b, true);

        assert_eq!(
            a,
            Encoding {
                ids: vec![1, 2],
                type_ids: vec![0, 1],
                tokens: vec![String::from("Hello "), String::from("World!")],
                words: vec![0, 1],
                offsets: vec![(0, 6), (6, 12)],
                special_tokens_mask: vec![0, 0],
                attention_mask: vec![1, 1],
                overflowing: vec![],
            }
        );
    }

    #[test]
    fn truncate() {
        let mut a = Encoding {
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec![
                String::from("Hello"),
                String::from("World"),
                String::from("!"),
            ],
            words: vec![0, 1, 2],
            offsets: vec![(0, 5), (6, 11), (11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            overflowing: vec![],
        };
        a.truncate(2, 0);

        assert_eq!(
            a,
            Encoding {
                ids: vec![1, 2],
                type_ids: vec![0, 0],
                tokens: vec![String::from("Hello"), String::from("World")],
                words: vec![0, 1],
                offsets: vec![(0, 5), (6, 11)],
                special_tokens_mask: vec![0, 0],
                attention_mask: vec![1, 1],
                overflowing: vec![Encoding {
                    ids: vec![3],
                    type_ids: vec![0],
                    tokens: vec![String::from("!")],
                    words: vec![2],
                    offsets: vec![(11, 12)],
                    special_tokens_mask: vec![0],
                    attention_mask: vec![1],
                    overflowing: vec![],
                }]
            }
        );
    }

    #[test]
    fn word_boundaries() {
        let encoding = Encoding {
            words: vec![0, 0, 1, 1, 1, 2, 3, 4, 4],
            ..Default::default()
        };
        assert_eq!(encoding.word_boundaries(0), Some((0, 2)));
        assert_eq!(encoding.word_boundaries(1), Some((0, 2)));
        assert_eq!(encoding.word_boundaries(2), Some((2, 5)));
        assert_eq!(encoding.word_boundaries(3), Some((2, 5)));
        assert_eq!(encoding.word_boundaries(5), Some((5, 6)));
        assert_eq!(encoding.word_boundaries(8), Some((7, 9)));
    }

    #[test]
    fn mappings() {
        let encoding = Encoding {
            words: vec![0, 0, 1, 1, 1, 2, 3, 4, 4],
            offsets: vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 10),
                (10, 12),
            ],
            ..Default::default()
        };
        assert_eq!(encoding.char_to_token_offsets(0), Some((0, 1)));
        assert_eq!(encoding.char_to_token_offsets(1), Some((1, 2)));
        assert_eq!(encoding.char_to_token_offsets(2), Some((2, 3)));
        assert_eq!(encoding.char_to_token_offsets(8), Some((7, 10)));
        assert_eq!(encoding.char_to_token_offsets(10), Some((10, 12)));
        assert_eq!(encoding.char_to_word_offsets(0), Some((0, 2)));
        assert_eq!(encoding.char_to_word_offsets(1), Some((0, 2)));
        assert_eq!(encoding.char_to_word_offsets(2), Some((2, 5)));
        assert_eq!(encoding.char_to_word_offsets(8), Some((7, 12)));
        assert_eq!(encoding.char_to_word_offsets(10), Some((7, 12)));
        assert_eq!(encoding.token_to_word_offsets(0), Some((0, 2)));
        assert_eq!(encoding.token_to_word_offsets(1), Some((0, 2)));
        assert_eq!(encoding.token_to_word_offsets(2), Some((2, 5)));
        assert_eq!(encoding.token_to_word_offsets(3), Some((2, 5)));
        assert_eq!(encoding.token_to_word_offsets(7), Some((7, 12)));
        assert_eq!(encoding.token_to_word_offsets(8), Some((7, 12)));
    }
}
