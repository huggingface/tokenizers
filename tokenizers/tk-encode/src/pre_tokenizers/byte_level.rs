use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

use crate::tokenizer::Encoding;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Provides all the necessary steps to handle the BPE tokenization at the byte-level. Takes care
/// of all the required processing steps to transform a UTF-8 string as needed before and after the
/// BPE model does its job.
#[non_exhaustive]
pub struct ByteLevel {
    /// Whether to add a leading space to the first word. This allows to treat the leading word
    /// just as any other word.
    pub add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    pub trim_offsets: bool,

    /// Whether to use the standard GPT2 regex for whitespace splitting
    /// Set it to False if you want to use your own splitting.
    ///
    /// The one pre-tokenizer field with a serde default, and it is `true`: configs written before
    /// `use_regex` existed have to keep loading with the regex on.
    pub use_regex: bool,
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl ByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    pub fn alphabet() -> [char; 256] {
        *BYTES_CHAR_LOOKUP
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

pub fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    encoding.process_tokens_with_offsets_mut(|(i, (token, offsets))| {
        let mut leading_spaces = token
            .chars()
            .take_while(|c| *c == BYTES_CHAR_LOOKUP[b' ' as usize] || c.is_whitespace())
            .count();
        let trailing_spaces = token
            .chars()
            .rev()
            .take_while(|c| *c == BYTES_CHAR_LOOKUP[b' ' as usize] || c.is_whitespace())
            .count();

        if leading_spaces > 0 || trailing_spaces > 0 {
            if leading_spaces > 0 {
                // If user uses `is_pretokenized=True` we might have
                // offsets that might begin at the start of the string but are
                // NOT the first token.
                let is_first = i == 0 || offsets.0 == 0;
                if is_first && add_prefix_space && leading_spaces == 1 {
                    // If we are processing the first pair of offsets, with `add_prefix_space`,
                    // then we shouldn't remove anything we added. If there are more than one
                    // leading spaces though, it means we didn't add them, and they should be
                    // removed.
                    leading_spaces = 0;
                }
                offsets.0 = std::cmp::min(offsets.0 + leading_spaces, offsets.1);
            }
            if trailing_spaces > 0 && offsets.1 >= trailing_spaces {
                offsets.1 = std::cmp::max(offsets.1 - trailing_spaces, offsets.0);
            }
        }
    });
}
