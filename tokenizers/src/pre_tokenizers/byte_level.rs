use crate::tokenizer::{Decoder, PreTokenizer};
use onig::Regex;
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
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .unwrap();
    static ref BYTES_CHAR: HashMap<u8, u32> = bytes_char();
    static ref CHAR_BYTES: HashMap<u32, u8> =
        bytes_char().into_iter().map(|(c, b)| (b, c)).collect();
}

pub struct ByteLevel;
impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, s: &str) -> Vec<String> {
        RE.find_iter(s)
            .map(|(start, end)| s[start..end].to_owned())
            .map(|s| {
                s.into_bytes()
                    .iter()
                    .map(|b| std::char::from_u32(BYTES_CHAR[b]).unwrap())
                    .collect()
            })
            .collect()
    }
}

impl Decoder for ByteLevel {
    fn decode(&self, tokens: Vec<String>) -> String {
        String::from_utf8_lossy(
            &tokens
                .join("")
                .chars()
                .map(|c| CHAR_BYTES[&(c as u32)])
                .collect::<Vec<_>>(),
        )
        .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::ByteLevel;
    use crate::tokenizer::{Decoder, PreTokenizer};

    #[test]
    fn pre_tokenization() {
        let pre_tok = ByteLevel;
        assert_eq!(
            pre_tok.pre_tokenize("Hello my friend, how is your day going?"),
            vec![
                "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing", "?"
            ]
        );
    }

    #[test]
    fn decoding() {
        let decoder = ByteLevel;
        assert_eq!(
            "Hello my friend, how is your day going?",
            decoder.decode(
                vec![
                    "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing", "?"
                ]
                .into_iter()
                .map(|s| s.into())
                .collect::<Vec<String>>()
            )
        );
    }

    #[test]
    fn decode_works_on_separated_tokens() {
        let samples = vec![
            String::from(
                "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ \"",
            ),
            String::from("An equal number have descenders , like p or q in English : გ , დ , ე , ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც"),
        ];

        let bl = ByteLevel;
        for sample in samples {
            let pre_tokenized = bl.pre_tokenize(&sample);
            let separated_tokens = pre_tokenized
                .into_iter()
                .map(|token| token.split("").map(|t| t.into()).collect::<Vec<_>>())
                .flatten()
                .collect::<Vec<_>>();
            assert_eq!(sample, bl.decode(separated_tokens));
        }
    }
}
