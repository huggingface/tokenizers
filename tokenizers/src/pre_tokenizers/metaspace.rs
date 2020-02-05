use crate::tokenizer::{Decoder, Offsets, PreTokenizer, Result};

/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
pub struct Metaspace {
    replacement: char,
    add_prefix_space: bool,
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            replacement,
            add_prefix_space,
        }
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', true)
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<(String, Offsets)>> {
        let s = if self.add_prefix_space && !s.starts_with(' ') {
            format!(" {}", s)
        } else {
            s.to_owned()
        };

        let mut words = vec![];
        let mut word = Vec::with_capacity(1000);
        let mut offset = 0;
        s.chars().for_each(|c| {
            if c.is_whitespace() {
                if !word.is_empty() {
                    let offsets = (offset - word.len(), offset);
                    words.push((word.drain(0..).collect::<String>(), offsets));
                }
                word.push(self.replacement)
            } else {
                word.push(c);
            }
            offset += 1;
        });
        if !word.is_empty() {
            let offsets = (offset - word.len(), offset);
            words.push((word.drain(0..).collect::<String>(), offsets));
        }

        Ok(words)
    }
}

impl Decoder for Metaspace {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(tokens
            .iter()
            .flat_map(|t| t.chars())
            .enumerate()
            .map(|(i, c)| {
                if c == self.replacement {
                    if i == 0 && self.add_prefix_space {
                        None
                    } else {
                        Some(' ')
                    }
                } else {
                    Some(c)
                }
            })
            .filter(|c| c.is_some())
            .map(|c| c.unwrap())
            .collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let pretok = Metaspace::new('▁', true);
        let res = pretok.pre_tokenize("Hey friend!").unwrap();
        assert_eq!(
            &res,
            &[("▁Hey".into(), (0, 4)), ("▁friend!".into(), (4, 12)),]
        );
    }

    #[test]
    fn multiple_spaces() {
        let pretok = Metaspace::new('▁', true);
        let res = pretok.pre_tokenize("Hey   friend!").unwrap();
        assert_eq!(
            &res,
            &[
                ("▁Hey".into(), (0, 4)),
                ("▁".into(), (4, 5)),
                ("▁".into(), (5, 6)),
                ("▁friend!".into(), (6, 14)),
            ]
        );
    }

    #[test]
    fn decode() {
        let decoder = Metaspace::new('▁', true);
        let res = decoder
            .decode(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(&res, "Hey friend!")
    }
}
