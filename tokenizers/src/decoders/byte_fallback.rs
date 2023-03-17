use crate::tokenizer::{Decoder, Result};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize, Default)]
/// Allows decoding Original BPE by joining all the tokens and then replacing
/// the suffix used to identify end-of-words by whitespaces
#[serde(tag = "type")]
#[non_exhaustive]
pub struct ByteFallback {}

impl ByteFallback {
    pub fn new() -> Self {
        Self {}
    }
}

impl Decoder for ByteFallback {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let mut new_tokens: Vec<String> = vec![];
        let mut previous_byte_tokens: Vec<u8> = vec![];

        for token in tokens {
            let bytes = if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                    Some(byte)
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(bytes) = bytes {
                previous_byte_tokens.push(bytes);
            } else {
                if !previous_byte_tokens.is_empty() {
                    if let Ok(string) = String::from_utf8(previous_byte_tokens.clone()) {
                        new_tokens.push(string.into());
                    } else {
                        for _ in 0..previous_byte_tokens.len() {
                            new_tokens.push("�".into());
                        }
                    }
                    previous_byte_tokens.clear();
                }
                new_tokens.push(token);
            }
        }
        if !previous_byte_tokens.is_empty() {
            if let Ok(string) = String::from_utf8(previous_byte_tokens.clone()) {
                new_tokens.push(string.into());
            } else {
                for _ in 0..previous_byte_tokens.len() {
                    new_tokens.push("�".into());
                }
            }
        }

        Ok(new_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = ByteFallback::new();
        let res = decoder
            .decode_chain(vec!["Hey".into(), "friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", "friend!"]);

        let res = decoder.decode_chain(vec!["<0x61>".into()]).unwrap();
        assert_eq!(res, vec!["a"]);

        let res = decoder.decode_chain(vec!["<0xE5>".into()]).unwrap();
        assert_eq!(res, vec!["�"]);

        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8f>".into()])
            .unwrap();
        assert_eq!(res, vec!["�", "�"]);

        // 叫
        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8f>".into(), "<0xab>".into()])
            .unwrap();
        assert_eq!(res, vec!["叫"]);

        let res = decoder
            .decode_chain(vec![
                "<0xE5>".into(),
                "<0x8f>".into(),
                "<0xab>".into(),
                "a".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["叫", "a"]);

        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8f>".into(), "a".into()])
            .unwrap();
        assert_eq!(res, vec!["�", "�", "a"]);
    }
}
