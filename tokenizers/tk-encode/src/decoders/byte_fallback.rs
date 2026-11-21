use crate::tokenizer::{Decode, Result};

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

/// Decodes a run of byte tokens, keeping every valid UTF-8 sequence intact and replacing
/// only the bytes that cannot start or continue one. The previous version decoded the run
/// as a whole, so a single stray byte after a valid character turned the entire run into
/// one replacement character per byte.
fn push_decoded(bytes: &[u8], out: &mut Vec<String>) {
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(text) => {
                if !text.is_empty() {
                    out.push(text.to_owned());
                }
                return;
            }
            Err(err) => {
                let valid = &rest[..err.valid_up_to()];
                if !valid.is_empty() {
                    // from_utf8 just reported this prefix as valid
                    out.push(
                        std::str::from_utf8(valid)
                            .expect("valid_up_to gives a valid prefix")
                            .to_owned(),
                    );
                }
                let tail = &rest[err.valid_up_to()..];
                match err.error_len() {
                    // An invalid sequence: one replacement per byte, which is what this
                    // decoder has always returned for undecodable byte tokens.
                    Some(len) => {
                        for _ in 0..len {
                            out.push("�".into());
                        }
                        rest = &tail[len..];
                    }
                    // Truncated at the end of the run: every remaining byte is replaced and
                    // the next token may complete the character.
                    None => {
                        for _ in 0..tail.len() {
                            out.push("�".into());
                        }
                        return;
                    }
                }
            }
        }
    }
}

impl Decode for ByteFallback {
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
                    push_decoded(&previous_byte_tokens, &mut new_tokens);
                    previous_byte_tokens.clear();
                }
                new_tokens.push(token);
            }
        }
        if !previous_byte_tokens.is_empty() {
            push_decoded(&previous_byte_tokens, &mut new_tokens);
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

    #[test]
    fn keeps_valid_characters_before_an_invalid_byte() {
        let decoder = ByteFallback::new();
        // F0 A0 80 80 is U+20000, E4 is a lead byte with no continuation
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

    #[test]
    fn decodes_runs_around_an_invalid_sequence() {
        let decoder = ByteFallback::new();
        // valid "a", one invalid byte, valid "b"
        let res = decoder
            .decode_chain(vec!["<0x61>".into(), "<0xFF>".into(), "<0x62>".into()])
            .unwrap();
        assert_eq!(res, vec!["a", "�", "b"]);
    }

    #[test]
    fn keeps_multi_byte_characters_split_across_runs() {
        let decoder = ByteFallback::new();
        let res = decoder
            .decode_chain(vec![
                "<0xE5>".into(),
                "<0x8F>".into(),
                "<0xAB>".into(),
                "<0xE5>".into(),
                "<0x8F>".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["叫", "�", "�"]);
    }
}
