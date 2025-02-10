use crate::tokenizer::{Decoder, Result};

use compact_str::{CompactString, ToCompactString};
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
    fn decode_chain<T: ToCompactString>(
        &self,
        tokens: Vec<T>,
    ) -> Result<Vec<impl ToCompactString>> {
        Ok(tokens
            .into_iter()
            .map(|token| {
                let chars: Vec<char> = token.to_compact_string().chars().collect();

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

                let new_token: CompactString = chars[start_cut..stop_cut].iter().collect();
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
            .decode_chain(vec![
                "Hey".to_owned(),
                " friend!".to_owned(),
                "HHH".to_owned(),
            ])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["ey", " friend!", "HH"]
        );

        let decoder = Strip::new('y', 0, 1);
        let res = decoder
            .decode_chain(vec!["Hey".to_owned(), " friend!".to_owned()])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["He", " friend!"]
        );
    }
}
