use crate::tokenizer::{Decoder, Result};

#[derive(Clone, Debug, Default)]
/// Strip just removes the `content` when it appears at the start position up until the stop position.
#[non_exhaustive]
pub struct Strip {
    pub content: char,
    pub start: usize,
    pub stop: usize,
}

impl Strip {
    pub fn new(content: char, start: usize, stop: usize) -> Self {
        Self {
            content,
            start,
            stop,
        }
    }
}

impl Decoder for Strip {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .into_iter()
            .map(|token| {
                let chars: Vec<char> = token.chars().collect();

                let mut start_cut = 0;
                for (i, &c) in chars.iter().enumerate().take(self.start) {
                    if c == self.content {
                        start_cut = i + 1;
                        continue;
                    } else {
                        break;
                    }
                }

                let mut stop_cut = chars.len();
                for index in (start_cut..chars.len()).rev().take(self.stop) {
                    if chars[index] == self.content {
                        stop_cut = index;
                        continue;
                    } else {
                        break;
                    }
                }

                let new_token: String = chars[start_cut..stop_cut].iter().collect();
                new_token
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = Strip::new('H', 1, 0);
        let res = decoder
            .decode_chain(vec!["Hey".into(), " friend!".into(), "HHH".into()])
            .unwrap();
        assert_eq!(res, vec!["ey", " friend!", "HH"]);

        let decoder = Strip::new('y', 0, 1);
        let res = decoder
            .decode_chain(vec!["Hey".into(), " friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["He", " friend!"]);
    }

    #[test]
    fn decode_short_tokens() {
        // fewer chars than `stop`, all of them `content`
        let decoder = Strip::new('H', 0, 3);
        let res = decoder.decode_chain(vec!["".into(), "HH".into()]).unwrap();
        assert_eq!(res, vec!["", ""]);

        // the leading and trailing strips together ask for more chars than the token has
        let decoder = Strip::new('H', 2, 2);
        let res = decoder
            .decode_chain(vec!["H".into(), "HHH".into(), "HeH".into()])
            .unwrap();
        assert_eq!(res, vec!["", "", "e"]);
    }
}
