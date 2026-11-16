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

fn flush_byte_tokens(bytes: &[u8], out: &mut Vec<String>) {
    for chunk in bytes.utf8_chunks() {
        let valid = chunk.valid();
        if !valid.is_empty() {
            out.push(valid.to_owned());
        }
        for _ in chunk.invalid() {
            out.push("�".into());
        }
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
                    flush_byte_tokens(&previous_byte_tokens, &mut new_tokens);
                    previous_byte_tokens.clear();
                }
                new_tokens.push(token);
            }
        }
        if !previous_byte_tokens.is_empty() {
            flush_byte_tokens(&previous_byte_tokens, &mut new_tokens);
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

        // Valid 4-byte UTF-8 character followed by an invalid stray lead byte (#2625)
        let res = decoder
            .decode_chain(vec![
                "a".into(),
                "<0xF0>".into(),
                "<0xA0>".into(),
                "<0x80>".into(),
                "<0x80>".into(),
                "<0xE4>".into(),
                "ok".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["a", "𠀀", "�", "ok"]);
    }
}
