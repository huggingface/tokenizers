use crate::tokenizer::{Model, Result, Token};
use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Debug)]
pub enum Error {
    MissingUnkToken,
}
impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingUnkToken => write!(
                fmt,
                "WordPiece error: Missing [UNK] token from the vocabulary"
            ),
        }
    }
}

pub struct WordPiece {
    unk_token: String,
    max_input_chars_per_word: usize,
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
}

impl Default for WordPiece {
    fn default() -> Self {
        WordPiece {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            unk_token: String::from("[UNK]"),
            max_input_chars_per_word: 100,
        }
    }
}

impl WordPiece {
    pub fn from_files(
        vocab: &str,
        unk_token: String,
        max_input_chars_per_word: Option<usize>,
    ) -> std::io::Result<Self> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(WordPiece {
            vocab: vocab.clone(),
            vocab_r: vocab.into_iter().map(|(token, id)| (id, token)).collect(),
            unk_token,
            max_input_chars_per_word: max_input_chars_per_word.unwrap_or(100),
        })
    }
}

impl Model for WordPiece {
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<String>) -> Result<Vec<Token>> {
        let mut output_tokens = vec![];

        let mut offset = 0usize;
        for token in sentence {
            let char_len = token.chars().count();
            if char_len > self.max_input_chars_per_word {
                output_tokens.push(Token {
                    value: self.unk_token.clone(),
                    id: *self
                        .vocab
                        .get(&self.unk_token)
                        .ok_or(Error::MissingUnkToken)?,
                    offsets: (offset, offset + char_len),
                });
                continue;
            }

            let mut is_bad = false;
            let mut start = 0;
            let mut sub_tokens: Vec<Token> = vec![];
            let chars = token.chars().collect::<Vec<_>>();

            while start < chars.len() {
                let mut end = chars.len();
                let mut cur_str = None;

                while start < end {
                    let mut substr = chars[start..end].iter().collect::<String>();
                    if start > 0 {
                        substr = format!("##{}", substr);
                    }
                    if self.vocab.contains_key(&substr) {
                        cur_str = Some(Token {
                            id: self.vocab[&substr],
                            value: substr,
                            offsets: (offset + start, offset + end),
                        });
                        break;
                    }
                    end -= 1;
                }

                if cur_str.is_none() {
                    is_bad = true;
                    break;
                }

                sub_tokens.push(cur_str.unwrap());
                start = end;
            }

            if is_bad {
                output_tokens.push(Token {
                    value: self.unk_token.clone(),
                    id: *self
                        .vocab
                        .get(&self.unk_token)
                        .ok_or(Error::MissingUnkToken)?,
                    offsets: (offset, offset + char_len),
                });
            } else {
                output_tokens.extend(sub_tokens);
            }

            offset += char_len;
        }

        Ok(output_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }
}
