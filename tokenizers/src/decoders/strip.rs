use crate::tokenizer::{Decoder, Result};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize, Default)]
/// Strip is a simple trick which converts tokens looking like `<0x61>`
/// to pure bytes, and attempts to make them into a string. If the tokens
/// cannot be decoded you will get � instead for each inconvertible byte token
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
                // Bound the trailing strip by the token length: `self.stop` is a raw
                // config value, so without `.min(chars.len())` a token made entirely
                // of `content` and shorter than `stop` keeps decrementing `index` past
                // 0, underflowing `usize` (debug panic / out-of-bounds index in release).
                for i in 0..self.stop.min(chars.len()) {
                    let index = chars.len() - i - 1;
                    if chars[index] == self.content {
                        stop_cut = index;
                        continue;
                    } else {
                        break;
                    }
                }

                // The leading and trailing windows can overlap when a short token is
                // stripped from both ends (start_cut > stop_cut), which would make the
                // slice range reversed and panic. Clamp so an over-stripped token
                // collapses to an empty string instead.
                let stop_cut = stop_cut.max(start_cut);
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
    fn short_all_content_token_does_not_panic() {
        // A token made entirely of `content` and shorter than `stop` used to
        // underflow `chars.len() - i - 1` (debug: subtract overflow; release:
        // out-of-bounds index). It should strip what is there and stop.
        let decoder = Strip::new('_', 0, 2);
        let res = decoder.decode_chain(vec!["_".into()]).unwrap();
        assert_eq!(res, vec![""]);

        let decoder = Strip::new('a', 0, 5);
        let res = decoder
            .decode_chain(vec!["aa".into(), "aab".into()])
            .unwrap();
        assert_eq!(res, vec!["", "aab"]);
    }

    #[test]
    fn overlapping_start_stop_windows_collapse_to_empty() {
        // When the leading and trailing windows overlap on a short token,
        // start_cut can exceed stop_cut; the slice used to panic on a reversed
        // range and now collapses to an empty string.
        let decoder = Strip::new('H', 2, 1);
        let res = decoder.decode_chain(vec!["HH".into()]).unwrap();
        assert_eq!(res, vec![""]);
    }
}
