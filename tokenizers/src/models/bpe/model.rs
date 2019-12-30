use super::{Cache, Error, Pair, Word};
use crate::tokenizer::{Model, Offsets, Result, Token};
use rand::{thread_rng, Rng};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
};

pub struct BPE {
    /// The vocabulary assigns a number to each token.
    vocab: HashMap<String, u32>,
    /// Reversed vocabulary, to rebuild sentences.
    vocab_r: HashMap<u32, String>,
    /// Contains the mapping between Pairs and their (rank, new_id).
    merges: HashMap<Pair, (u32, u32)>,
    /// Contains the cache for optimizing the encoding step.
    cache: Cache<String, Word>,
    /// Dropout probability for merges. 0 = no dropout is the default. At 1.0, tokenization will
    /// perform no merges, so the result will just be characters.
    dropout: Option<f32>,
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
            dropout: None,
        }
    }

    /// Initialize a BPE model with [dropout](https://arxiv.org/abs/1910.13267).
    pub fn with_dropout(
        vocab: HashMap<String, u32>,
        vocab_r: HashMap<u32, String>,
        merges: HashMap<Pair, (u32, u32)>,
        dropout: f32,
    ) -> Result<Self> {
        if dropout < 0.0 || dropout > 1.0 {
            Err(Error::InvalidDropout.into())
        } else {
            Ok(BPE {
                vocab,
                vocab_r,
                merges,
                cache: Cache::new(),
                dropout: if dropout == 0.0 { None } else { Some(dropout) },
            })
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
            dropout: None,
        })
    }
}

impl Model for BPE {
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<(String, Offsets)>) -> Result<Vec<Token>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }

        let mut encoded: Vec<Token> = Vec::with_capacity(sentence.len());
        let mut cached_words = self.cache.get_values(
            &sentence
                .iter()
                .map(|(s, _)| s.to_owned())
                .collect::<Vec<_>>(),
        );

        for (i, (w, initial_offsets)) in sentence.iter().enumerate() {
            // If we're using dropout or we don't have a cache hit, we have to compute
            // merges for this word.
            if self.dropout.is_some() || cached_words[i].is_none() {
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
                                .map(|rank| {
                                    if let Some(dropout) = self.dropout {
                                        // With probability `dropout` we'll ignore
                                        // this merge.
                                        if thread_rng().gen::<f32>() < dropout {
                                            &(std::u32::MAX, std::u32::MAX)
                                        } else {
                                            rank
                                        }
                                    } else {
                                        rank
                                    }
                                })
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

            let word = cached_words[i].as_ref().unwrap();
            let tokens = word
                .get_chars()
                .iter()
                .zip(word.get_offsets())
                .map(|(id, offsets)| {
                    Token::new(
                        *id,
                        self.vocab_r[id].clone(),
                        (initial_offsets.0 + offsets.0, initial_offsets.0 + offsets.1),
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
            .map(|(k, v)| (k.0, v.unwrap()))
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
    // Test tokenization. With dropout set to 0 tokenization is deterministic,
    // so we know exactly what the result should be.
    //
    // To test this, we'll build a simple model to tokenize the word 'unrelated'.
    fn test_tokenize_with_and_without_dropout() {
        let vocab: HashMap<String, u32> = [
            ("u".into(), 0),
            ("n".into(), 1),
            ("r".into(), 2),
            ("e".into(), 3),
            ("l".into(), 4),
            ("a".into(), 5),
            ("t".into(), 6),
            ("d".into(), 7),
            ("re".into(), 8),
            ("at".into(), 9),
            ("ed".into(), 10),
            ("un".into(), 11),
            ("ated".into(), 12),
            ("rel".into(), 13),
            ("related".into(), 14),
            ("unrelated".into(), 15),
        ]
        .iter()
        .cloned()
        .collect();
        let vocab_r: HashMap<u32, String> = vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        let merges: HashMap<Pair, (u32, u32)> = [
            ((vocab["r"], vocab["e"]), (1u32, vocab["re"])), // 'r-e' -> 're'
            ((vocab["a"], vocab["t"]), (2u32, vocab["at"])), // 'a-t' -> 'at'
            ((vocab["e"], vocab["d"]), (3u32, vocab["ed"])), // 'e-d' -> 'ed'
            ((vocab["u"], vocab["n"]), (4u32, vocab["un"])), // 'u-n' -> 'un'
            ((vocab["at"], vocab["ed"]), (5u32, vocab["ated"])), // 'at-ed' -> 'ated'
            ((vocab["re"], vocab["l"]), (6u32, vocab["rel"])), // 're-l' -> 'rel'
            ((vocab["rel"], vocab["ated"]), (7u32, vocab["related"])), // 'rel-ated' -> 'related'
            ((vocab["un"], vocab["related"]), (8u32, vocab["unrelated"])), // 'un-related' -> 'unrelated'
        ]
        .iter()
        .cloned()
        .collect();
        let mut bpe = BPE::new(vocab, vocab_r, merges);

        let sentence: Vec<(String, Offsets)> = vec![("unrelated".into(), (0, 9))];

        // With no dropout:
        let tokens = bpe.tokenize(sentence.clone()).unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // Now set dropout to 1.0. Result should be no merges performed.
        bpe.dropout = Some(1.0);
        let tokens = bpe.tokenize(sentence.clone()).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "u".into(), (0, 1)),
                Token::new(1u32, "n".into(), (1, 2)),
                Token::new(2u32, "r".into(), (2, 3)),
                Token::new(3u32, "e".into(), (3, 4)),
                Token::new(4u32, "l".into(), (4, 5)),
                Token::new(5u32, "a".into(), (5, 6)),
                Token::new(6u32, "t".into(), (6, 7)),
                Token::new(3u32, "e".into(), (7, 8)),
                Token::new(7u32, "d".into(), (8, 9)),
            ]
        );

        // Now try with dropout between 0 and 1.
        bpe.dropout = Some(0.5);
        let tokens = bpe.tokenize(sentence).unwrap();
        assert!(!tokens.is_empty() && tokens.len() <= 9);
    }

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
        let result = BPE::from_files(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        );
        assert!(result.is_ok());

        let bpe = result.unwrap();

        // Check merges.
        assert_eq!(bpe.merges.get(&(0u32, 1u32)).unwrap(), &(1u32, 3u32));

        // Check vocab.
        assert_eq!(bpe.vocab.get("a").unwrap(), &0u32);
        assert_eq!(bpe.vocab.get("b").unwrap(), &1u32);
        assert_eq!(bpe.vocab.get("c").unwrap(), &2u32);
        assert_eq!(bpe.vocab.get("ab").unwrap(), &3u32);
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
