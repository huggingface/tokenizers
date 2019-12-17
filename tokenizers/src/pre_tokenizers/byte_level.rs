use crate::tokenizer::{Decoder, PreTokenizer, Result};
use regex::Regex;
use std::collections::HashMap;

fn bytes_char() -> HashMap<u8, u32> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    bs.into_iter().zip(cs).collect()
}

lazy_static! {
    static ref RE: Regex =
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
    static ref BYTES_CHAR: HashMap<u8, u32> = bytes_char();
    static ref CHAR_BYTES: HashMap<u32, u8> =
        bytes_char().into_iter().map(|(c, b)| (b, c)).collect();
}

pub struct ByteLevel {
    add_prefix_space: bool,
}
impl ByteLevel {
    pub fn new(add_prefix_space: bool) -> Self {
        ByteLevel { add_prefix_space }
    }
}

impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<String>> {
        let s = if self.add_prefix_space && !s.starts_with(' ') {
            format!(" {}", s)
        } else {
            s.to_owned()
        };

        Ok(RE
            .captures_iter(&s)
            .map(|capture| {
                let capture = capture.get(0).unwrap();
                let start = capture.start();
                let end = capture.end();

                // if our last character is a whitespace, followed by a non whitespace,
                // we don't want to return it
                let last = s[start..end].chars().last();
                let next = s[end..].chars().nth(0);
                if last.is_some()
                    && last.unwrap().is_whitespace()
                    && next.is_some()
                    && !next.unwrap().is_whitespace()
                {
                    if let Some(newstr) = s[start..end]
                        .chars()
                        .collect::<Vec<_>>()
                        .split_last()
                        .map(|(_, rest)| rest)
                        .map(|chars| chars.iter().collect::<String>())
                    {
                        return newstr;
                    }
                }
                // if our first char is not a whitespace but the previous one was, we return
                // a whitespace before our match
                let prev = s[0..start].chars().last();
                let current = s[start..end].chars().nth(0).map(|c| c.is_whitespace());
                if prev.is_some()
                    && prev.unwrap().is_whitespace()
                    && current.is_some()
                    && !current.unwrap()
                {
                    return format!(" {}", s[start..end].to_owned());
                }

                s[start..end].to_owned()
            })
            .map(|s| {
                s.into_bytes()
                    .iter()
                    .map(|b| std::char::from_u32(BYTES_CHAR[b]).unwrap())
                    .collect()
            })
            .collect())
    }
}

impl Decoder for ByteLevel {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        Ok(String::from_utf8_lossy(
            &tokens
                .join("")
                .chars()
                .map(|c| CHAR_BYTES[&(c as u32)])
                .collect::<Vec<_>>(),
        )
        .into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::ByteLevel;
    use crate::tokenizer::{Decoder, PreTokenizer};

    #[test]
    fn pre_tokenization() {
        let pre_tok = ByteLevel::new(false);
        assert_eq!(
            pre_tok
                .pre_tokenize("Hello my friend, how is your day going?")
                .unwrap(),
            vec![
                "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing", "?"
            ]
        );
    }

    #[test]
    fn decoding() {
        let decoder = ByteLevel::new(false);
        assert_eq!(
            "Hello my friend, how is your day going?",
            decoder
                .decode(
                    vec![
                        "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing",
                        "?"
                    ]
                    .into_iter()
                    .map(|s| s.into())
                    .collect::<Vec<String>>()
                )
                .unwrap()
        );
    }

    #[test]
    fn add_prefix_space() {
        let pre_tok = ByteLevel::new(true);
        assert_eq!(
            pre_tok
                .pre_tokenize("Hello my friend, how is your day going?")
                .unwrap(),
            vec![
                "ĠHello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing", "?"
            ]
        );
    }

    #[test]
    fn decode_works_on_separated_tokens() {
        let samples = vec![
            String::from(
                "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ \"",
            ),
            String::from(
                "An equal number have descenders , like p or q in English \
                 : გ , დ , ე , ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც",
            ),
        ];

        let bl = ByteLevel::new(false);
        for sample in samples {
            let pre_tokenized = bl.pre_tokenize(&sample).unwrap();
            let separated_tokens = pre_tokenized
                .into_iter()
                .map(|token| token.split("").map(|t| t.into()).collect::<Vec<_>>())
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(sample, bl.decode(separated_tokens).unwrap());
        }
    }
}
