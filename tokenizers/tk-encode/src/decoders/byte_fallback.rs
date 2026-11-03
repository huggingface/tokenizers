use crate::tokenizer::{Decoder, Result};

#[derive(Clone, Debug, Default)]
/// ByteFallback is a simple trick which converts tokens looking like `<0x61>`
/// to pure bytes, and attempts to make them into a string. If the tokens
/// cannot be decoded you will get � instead for each inconvertible byte token
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
                u8::from_str_radix(&token[3..5], 16).ok()
            } else {
                None
            };
            if let Some(bytes) = bytes {
                previous_byte_tokens.push(bytes);
            } else {
                if !previous_byte_tokens.is_empty() {
                    decode_bytes(&previous_byte_tokens, &mut new_tokens);
                    previous_byte_tokens.clear();
                }
                new_tokens.push(token);
            }
        }
        if !previous_byte_tokens.is_empty() {
            decode_bytes(&previous_byte_tokens, &mut new_tokens);
        }

        Ok(new_tokens)
    }
}

fn decode_bytes(bytes: &[u8], tokens: &mut Vec<String>) {
    for chunk in bytes.utf8_chunks() {
        if !chunk.valid().is_empty() {
            tokens.push(chunk.valid().to_owned());
        }
        // Preserve one replacement token per invalid byte.
        tokens.extend(chunk.invalid().iter().map(|_| "�".to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_mixed_valid_and_invalid_bytes() {
        let decoder = ByteFallback::new();
        let cases: &[(&[u8], &[&str])] = &[
            (&[0xF0, 0xA0, 0x80, 0x80, 0xE4], &["𠀀", "�"]),
            (&[0xE4, 0xF0, 0xA0, 0x80, 0x80], &["�", "𠀀"]),
            (&[0x61, 0xFF, 0x62], &["a", "�", "b"]),
            (&[0xE5, 0x8F, 0x61], &["�", "�", "a"]),
            (&[0xED, 0xA0, 0x80], &["�", "�", "�"]),
            (&[0xC0, 0xAF], &["�", "�"]),
        ];

        for &(bytes, expected) in cases {
            for suffix in [None, Some("ok")] {
                let mut tokens: Vec<String> =
                    bytes.iter().map(|byte| format!("<0x{byte:02X}>")).collect();
                let mut expected: Vec<String> =
                    expected.iter().map(|token| (*token).to_owned()).collect();
                if let Some(suffix) = suffix {
                    tokens.push(suffix.to_owned());
                    expected.push(suffix.to_owned());
                }
                assert_eq!(decoder.decode_chain(tokens).unwrap(), expected);
            }
        }
    }

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
