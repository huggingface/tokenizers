use crate::tokenizer::{Decoder, Result};
use compact_str::{CompactString, ToCompactString};
use monostate::MustBe;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Debug, Serialize, Default)]
/// ByteFallback is a simple trick which converts tokens looking like `<0x61>`
/// to pure bytes, and attempts to make them into a string. If the tokens
/// cannot be decoded you will get � instead for each inconvertible byte token
#[non_exhaustive]
pub struct ByteFallback {
    #[serde(rename = "type")]
    type_: MustBe!("ByteFallback"),
}

impl ByteFallback {
    pub fn new() -> Self {
        Self {
            type_: MustBe!("ByteFallback"),
        }
    }
}

impl Decoder for ByteFallback {
    fn decode_chain<T: ToCompactString>(
        &self,
        tokens: Vec<T>,
    ) -> Result<Vec<impl ToCompactString>> {
        let mut new_tokens: Vec<CompactString> = vec![];
        let mut previous_byte_tokens: Vec<u8> = vec![];

        for token in tokens {
            let token: CompactString = token.to_compact_string();
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
                    if let Ok(string) = CompactString::from_utf8(previous_byte_tokens.clone()) {
                        new_tokens.push(string);
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
            if let Ok(string) = CompactString::from_utf8(previous_byte_tokens.clone()) {
                new_tokens.push(string);
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
            .decode_chain(vec!["Hey".to_owned(), "friend!".to_owned()])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["Hey".to_owned(), "friend!".to_owned()]
        );

        let res = decoder.decode_chain(vec!["<0x61>".to_owned()]).unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["a".to_owned()]
        );

        let res = decoder.decode_chain(vec!["<0xE5>".to_owned()]).unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["�"]
        );

        let res = decoder
            .decode_chain(vec!["<0xE5>".to_owned(), "<0x8f>".to_owned()])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["�".to_owned(), "�".to_owned()]
        );

        // 叫
        let res = decoder
            .decode_chain(vec![
                "<0xE5>".to_owned(),
                "<0x8f>".to_owned(),
                "<0xab>".to_owned(),
            ])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["叫"]
        );

        let res = decoder
            .decode_chain(vec![
                "<0xE5>".to_owned(),
                "<0x8f>".to_owned(),
                "<0xab>".to_owned(),
                "a".to_owned(),
            ])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["叫".to_owned(), "a".to_owned()]
        );

        let res = decoder
            .decode_chain(vec![
                "<0xE5>".to_owned(),
                "<0x8f>".to_owned(),
                "a".to_owned(),
            ])
            .unwrap();
        assert_eq!(
            res.into_iter()
                .map(|t| t.to_compact_string())
                .collect::<Vec<_>>(),
            vec!["�".to_owned(), "�".to_owned(), "a".to_owned()]
        );
    }
}
