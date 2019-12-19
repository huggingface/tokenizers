use super::{Cache, Error, Pair, Word};
use crate::tokenizer::{Model, Result, Token};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
};

pub struct BPE {
    /// The vocabulary assigns a number to each token
    vocab: HashMap<String, u32>,
    /// Reversed vocabulary, to rebuild sentences
    vocab_r: HashMap<u32, String>,
    /// Contains the mapping between Pairs and their (rank, new_id)
    merges: HashMap<Pair, (u32, u32)>,
    /// Contains the cache for optimizing the encoding step
    cache: Cache<String, Word>,
}

impl BPE {
    pub fn new(
        vocab: HashMap<String, u32>,
        vocab_r: HashMap<u32, String>,
        merges: HashMap<Pair, (u32, u32)>,
    ) -> Self {
        BPE {
            vocab,
            vocab_r,
            merges,
            cache: Cache::new(),
        }
    }

    pub fn empty() -> Self {
        BPE::new(HashMap::new(), HashMap::new(), HashMap::new())
    }

    pub fn from_files(vocab: &str, merges: &str) -> Result<Self> {
        // Read vocab.json
        let vocab_file = File::open(vocab)?;
        let mut vocab_file = BufReader::new(vocab_file);

        let mut buffer = String::new();
        vocab_file.read_to_string(&mut buffer)?;
        let json: Value = serde_json::from_str(&buffer)?;
        let mut vocab = HashMap::new();
        match json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };

        // Read merges file
        let merge_file = File::open(merges)?;
        let merge_file = BufReader::new(merge_file);
        let mut merges = HashMap::<Pair, (u32, u32)>::new();
        for (rank, line) in merge_file.lines().enumerate() {
            let line = line?;
            if line.starts_with("#version") {
                // Skip line with: #version
                continue;
            }

            let parts = line.split(' ').collect::<Vec<_>>();
            if parts.len() != 2 {
                return Err(Error::BadMerges(rank + 1).into());
            }

            let a = vocab
                .get(parts[0])
                .ok_or_else(|| Error::MergeTokenOutOfVocabulary(parts[0].to_owned()))?;
            let b = vocab
                .get(parts[1])
                .ok_or_else(|| Error::MergeTokenOutOfVocabulary(parts[1].to_owned()))?;
            let pair = (*a, *b);
            let new_token = format!("{}{}", parts[0], parts[1]);
            let new_id = vocab
                .get(&new_token)
                .ok_or(Error::MergeTokenOutOfVocabulary(new_token))?;

            merges.insert(pair, (rank as u32, *new_id));
        }

        Ok(BPE {
            vocab: vocab.clone(),
            vocab_r: vocab.into_iter().map(|(token, id)| (id, token)).collect(),
            merges,
            cache: Cache::new(),
        })
    }
}

impl Model for BPE {
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<String>) -> Result<Vec<Token>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }

        let mut encoded: Vec<Token> = Vec::with_capacity(sentence.len());
        let mut cached_words = self.cache.get_values(&sentence);

        for (i, w) in sentence.iter().enumerate() {
            if cached_words[i].is_none() {
                let mut word = Word::new();
                for c in w.chars() {
                    match self.vocab.get(&c.to_string()) {
                        // TODO: Handle UNK
                        None => {
                            println!("{} is an unknown character. Skip it.", c.escape_unicode())
                        }
                        Some(id) => word.add(*id),
                    }
                }

                loop {
                    if word.get_chars().len() < 2 {
                        break;
                    }

                    let ((rank, new_id), pair) = word
                        .get_chars()
                        .windows(2)
                        .map(|window| {
                            let pair = (window[0], window[1]);
                            let rank = self
                                .merges
                                .get(&pair)
                                .unwrap_or(&(std::u32::MAX, std::u32::MAX));
                            (rank, pair)
                        })
                        .min()
                        .unwrap();

                    if *rank == std::u32::MAX {
                        // We are done merging this word
                        break;
                    }

                    // Let's merge
                    word.merge(pair.0, pair.1, *new_id);
                }

                cached_words[i] = Some(word);
            }

            // Offsets are word-based, we need to translate them to be sentence-based
            let last_offset = encoded.last().map(|token| token.offsets.1).unwrap_or(0);

            let word = cached_words[i].as_ref().unwrap();
            let tokens = word
                .get_chars()
                .iter()
                .zip(word.get_offsets())
                .map(|(id, offsets)| {
                    Token::new(
                        *id,
                        self.vocab_r[id].clone(),
                        (last_offset + offsets.0, last_offset + offsets.1),
                    )
                })
                .collect::<Vec<_>>();

            encoded.extend(tokens);
        }

        // Also update cache
        let (keys, values) = sentence
            .into_iter()
            .zip(cached_words)
            .filter(|(_, v)| v.is_some())
            .map(|(k, v)| (k, v.unwrap()))
            .unzip::<_, _, Vec<String>, Vec<Word>>();
        self.cache.set_values(keys, values);

        Ok(encoded)
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
    use tempfile::NamedTempFile;

    #[test]
    // Ensure `BPE::from_files` works as expected.
    fn test_bpe_from_files() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
            .unwrap();

        // Set up merges file.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file
            .write_all("#version: 0.2\na b".as_bytes())
            .unwrap();

        // Make sure we can instatiate a BPE model from the files.
        assert!(BPE::from_files(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap()
        )
        .is_ok());
    }

    #[test]
    // Ensure `MergeTokenOutOfVocabulary` error is returned when it should be.
    fn test_bpe_from_files_merge_token_oov() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
            .unwrap();

        // Set up merges file.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file
            .write_all("#version: 0.2\na b\na d".as_bytes())
            .unwrap();

        // Ensure the result of BPE::from_files is a MergeTokenOutOfVocabulary error.
        match BPE::from_files(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        ) {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::MergeTokenOutOfVocabulary(token)) => {
                    assert_eq!(*token, String::from("d"))
                }
                _ => unreachable!(),
            },
        }
    }

    #[test]
    // Ensure `BadMerges` error is returned when there is an invalid line in the
    // merges.txt file.
    fn test_bpe_from_files_bad_merges() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
            .unwrap();

        // Set up merges file with a bad line.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file
            .write_all("#version: 0.2\na b\nc".as_bytes())
            .unwrap();

        // Ensure the result of BPE::from_files is a BadMerges error.
        match BPE::from_files(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        ) {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::BadMerges(line)) => assert_eq!(*line, 3usize),
                _ => unreachable!(),
            },
        }
    }
}
