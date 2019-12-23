/// The various possible padding directions
#[derive(Debug, Clone)]
pub enum PaddingDirection {
    Left,
    Right,
}

/// The Encoding struct represents the output of the Tokenizer
#[derive(Default, PartialEq, Debug, Clone)]
pub struct Encoding {
    original: String,
    normalized: String,
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<(usize, usize)>,
    special_tokens_mask: Vec<u32>,
    attention_mask: Vec<u32>,
    overflowing: Option<Box<Encoding>>,
}
impl Encoding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        original: String,
        normalized: String,
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        offsets: Vec<(usize, usize)>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        overflowing: Option<Box<Encoding>>,
    ) -> Self {
        Encoding {
            original,
            normalized,
            ids,
            type_ids,
            tokens,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
        }
    }

    pub fn get_original(&self) -> &str {
        &self.original
    }

    pub fn get_normalized(&self) -> &str {
        &self.normalized
    }

    pub fn get_tokens(&self) -> &[String] {
        &self.tokens[..]
    }

    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    pub fn get_offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn get_overflowing(&self) -> Option<&Encoding> {
        self.overflowing.as_ref().map(|b| &**b)
    }

    pub fn take_overflowing(&mut self) -> Option<Box<Encoding>> {
        self.overflowing.take()
    }

    pub fn truncate(&mut self, max_len: usize, stride: usize) {
        if max_len >= self.ids.len() {
            return;
        }

        let mut o_ids = self.ids.split_off(max_len);
        let mut o_type_ids = self.type_ids.split_off(max_len);
        let mut o_tokens = self.tokens.split_off(max_len);
        let mut o_offsets = self.offsets.split_off(max_len);
        let mut o_spe_toks = self.special_tokens_mask.split_off(max_len);
        let mut o_attent = self.attention_mask.split_off(max_len);

        // Figure out offsets for original and normalized
        // TODO: We will be able to retrive the right part of original
        // only when we will have the alignment difference between both
        // For now we will use the normalized offset...
        let max = self
            .offsets
            .iter()
            .fold(0, |max, (_, end)| if *end > max { *end } else { max });
        let trunc_original = self.original.split_off(max);
        let trunc_normalized = self.normalized.split_off(max);

        if stride > 0 {
            o_ids = prepend_stride(&self.ids, o_ids, stride);
            o_type_ids = prepend_stride(&self.type_ids, o_type_ids, stride);
            o_tokens = prepend_stride(&self.tokens, o_tokens, stride);
            o_offsets = prepend_stride(&self.offsets, o_offsets, stride);
            o_spe_toks = prepend_stride(&self.special_tokens_mask, o_spe_toks, stride);
            o_attent = prepend_stride(&self.attention_mask, o_attent, stride);
        }

        self.overflowing = Some(Box::new(Encoding {
            original: trunc_original,
            normalized: trunc_normalized,
            ids: o_ids,
            type_ids: o_type_ids,
            tokens: o_tokens,
            offsets: o_offsets,
            special_tokens_mask: o_spe_toks,
            attention_mask: o_attent,
            overflowing: None,
        }));
    }

    pub fn merge_with(&mut self, pair: Encoding) {
        self.original.push_str(&pair.original);
        self.normalized.push_str(&pair.normalized);
        self.ids.extend(pair.ids);
        self.type_ids.extend(pair.type_ids);
        self.tokens.extend(pair.tokens);

        let starting_offset = self
            .offsets
            .iter()
            .fold(0, |max, (_, end)| if *end > max { *end } else { max });
        self.offsets.extend(
            pair.offsets
                .into_iter()
                .map(|(start, end)| (start + starting_offset, end + starting_offset))
                .collect::<Vec<_>>(),
        );
        self.special_tokens_mask.extend(pair.special_tokens_mask);
        self.attention_mask.extend(pair.attention_mask);
        // TODO: Handle the overflowing
    }

    pub fn pad(
        &mut self,
        pad_length: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        direction: &PaddingDirection,
    ) {
        if self.ids.len() > pad_length {
            // We just do nothing if the wanted padding length is smaller than us
            return;
        }
        let pad_length = pad_length - self.ids.len();

        let ids_pad = vec![pad_id; pad_length];
        let type_ids_pad = vec![pad_type_id; pad_length];
        let tokens_pad = vec![pad_token.to_owned(); pad_length];
        let attention_pad = vec![0; pad_length];
        let special_pad = vec![1; pad_length];
        let offsets_pad = vec![(0, 0); pad_length];

        match direction {
            PaddingDirection::Left => {
                self.ids = [&ids_pad[..], &self.ids[..]].concat();
                self.type_ids = [&type_ids_pad[..], &self.type_ids[..]].concat();
                self.tokens = [&tokens_pad[..], &self.tokens[..]].concat();
                self.attention_mask = [&attention_pad[..], &self.attention_mask[..]].concat();
                self.special_tokens_mask =
                    [&special_pad[..], &self.special_tokens_mask[..]].concat();
                self.offsets = [&offsets_pad[..], &self.offsets[..]].concat();
            }
            PaddingDirection::Right => {
                self.ids = [&self.ids[..], &ids_pad[..]].concat();
                self.type_ids = [&self.type_ids[..], &type_ids_pad[..]].concat();
                self.tokens = [&self.tokens[..], &tokens_pad[..]].concat();
                self.attention_mask = [&self.attention_mask[..], &attention_pad[..]].concat();
                self.special_tokens_mask =
                    [&self.special_tokens_mask[..], &special_pad[..]].concat();
                self.offsets = [&self.offsets[..], &offsets_pad[..]].concat();
            }
        }
    }
}

/// Prepend the `stride` last elements of the `previous` Vec to the current Vec
// A new Vec is instantiated though.
fn prepend_stride<T: Clone>(previous: &[T], current: Vec<T>, stride: usize) -> Vec<T> {
    let prev = previous
        .iter()
        .rev()
        .take(stride)
        .cloned()
        .rev()
        .collect::<Vec<_>>();

    [&prev[..], &current[..]].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepend_stride() {
        let prev = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let curr = vec![9, 10, 11, 12];

        assert_eq!(prepend_stride(&prev, curr, 3), vec![6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn merge_encodings() {
        let mut a = Encoding {
            original: String::from("Hello "),
            normalized: String::from("Hello "),
            ids: vec![1],
            type_ids: vec![0],
            tokens: vec![String::from("Hello ")],
            offsets: vec![(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            overflowing: None,
        };
        let b = Encoding {
            original: String::from("World!"),
            normalized: String::from("World!"),
            ids: vec![2],
            type_ids: vec![1],
            tokens: vec![String::from("World!")],
            offsets: vec![(0, 6)],
            special_tokens_mask: vec![0],
            attention_mask: vec![1],
            overflowing: None,
        };
        a.merge_with(b);

        assert_eq!(
            a,
            Encoding {
                original: String::from("Hello World!"),
                normalized: String::from("Hello World!"),
                ids: vec![1, 2],
                type_ids: vec![0, 1],
                tokens: vec![String::from("Hello "), String::from("World!")],
                offsets: vec![(0, 6), (6, 12)],
                special_tokens_mask: vec![0, 0],
                attention_mask: vec![1, 1],
                overflowing: None,
            }
        );
    }

    #[test]
    fn truncate() {
        let mut a = Encoding {
            original: String::from("Hello World!"),
            normalized: String::from("Hello World!"),
            ids: vec![1, 2, 3],
            type_ids: vec![0, 0, 0],
            tokens: vec![
                String::from("Hello"),
                String::from("World"),
                String::from("!"),
            ],
            offsets: vec![(0, 5), (6, 11), (11, 12)],
            special_tokens_mask: vec![0, 0, 0],
            attention_mask: vec![1, 1, 1],
            overflowing: None,
        };
        a.truncate(2, 0);

        assert_eq!(
            a,
            Encoding {
                original: String::from("Hello World"),
                normalized: String::from("Hello World"),
                ids: vec![1, 2],
                type_ids: vec![0, 0],
                tokens: vec![String::from("Hello"), String::from("World")],
                offsets: vec![(0, 5), (6, 11)],
                special_tokens_mask: vec![0, 0],
                attention_mask: vec![1, 1],
                overflowing: Some(Box::new(Encoding {
                    original: String::from("!"),
                    normalized: String::from("!"),
                    ids: vec![3],
                    type_ids: vec![0],
                    tokens: vec![String::from("!")],
                    offsets: vec![(11, 12)],
                    special_tokens_mask: vec![0],
                    attention_mask: vec![1],
                    overflowing: None,
                }))
            }
        );
    }
}
