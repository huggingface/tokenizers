use crate::{
    pipeline,
    tokenizer::{Decoder, Result},
};

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

impl pipeline::Decoder for Strip {
    fn decode_token(
        &self,
        _state: &mut pipeline::DecoderState,
        _token_id: u32,
        token_bytes: &[u8],
        decoded: &mut Vec<u8>,
    ) -> Result<()> {
        let mut pat_buf = [0u8; 4];
        let pat = self.content.encode_utf8(&mut pat_buf).as_bytes();
        let mut token = token_bytes;
        for _ in 0..self.start {
            match token.strip_prefix(pat) {
                Some(rest) => token = rest,
                None => break,
            }
        }
        for _ in 0..self.stop {
            match token.strip_suffix(pat) {
                Some(rest) => token = rest,
                None => break,
            }
        }
        decoded.extend_from_slice(token);
        Ok(())
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
