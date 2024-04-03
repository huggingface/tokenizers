use crate::tokenizer::{Decoder, Result};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize, Default)]
// Strip removes prefixes and suffixes (of length at most start/stop,
// respectively) which consist solely of the given character.
#[serde(tag = "type")]
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
                for i in 0..self.stop {
                    let index = chars.len() - i - 1;
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
}
