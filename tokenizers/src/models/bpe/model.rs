use super::{Cache, Error, Pair, Word};
use crate::tokenizer::{Model, Offsets, Result, Token};
use rand::{thread_rng, Rng};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[derive(Default)]
struct Config {
    vocab: Option<HashMap<String, u32>>,
    vocab_r: Option<HashMap<u32, String>>,
    merges: Option<HashMap<Pair, (u32, u32)>>,
    cache_capacity: Option<usize>,
    dropout: Option<f32>,
    unk_token: Option<u32>,
}

/// A `BpeBuilder` can be used to create a `BPE` model with a custom configuration.
#[derive(Default)]
pub struct BpeBuilder {
    config: Config,
}

impl BpeBuilder {
    /// Constructs a new `BpeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vocab (token -> ID) and merges mappings.
    pub fn vocab_and_merges(
        mut self,
        vocab: HashMap<String, u32>,
        merges: HashMap<Pair, (u32, u32)>,
    ) -> Self {
        self.config.vocab = Some(vocab);
        self.config.merges = Some(merges);
        self
    }

    /// Set the cache's capacity.
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.config.cache_capacity = Some(capacity);
        self
    }

    /// Use [dropout](https://arxiv.org/abs/1910.13267) with the model.
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = Some(dropout);
        self
    }

    /// Set the `UNK` token for the vocab.
    pub fn unk_token(mut self, unk_token: u32) -> Self {
        self.config.unk_token = Some(unk_token);
        self
    }

    /// Returns a `BPE` model that uses the `BpeBuilder`'s configuration.
    pub fn build(self) -> Result<BPE> {
        // Validate dropout.
        if let Some(p) = self.config.dropout {
            if p < 0.0 || p > 1.0 {
                return Err(Error::InvalidDropout.into());
            }
        }

        let vocab = self.config.vocab.unwrap_or_else(HashMap::new);
        let vocab_r = match self.config.vocab_r {
            Some(vocab_r) => vocab_r,
            None => vocab
                .iter()
                .map(|(key, val)| (*val, key.to_owned()))
                .collect(),
        };
        let merges = self.config.merges.unwrap_or_else(HashMap::new);
        let cache = match self.config.cache_capacity {
            Some(capacity) => Cache::new(capacity),
            None => Cache::default(),
        };

        Ok(BPE {
            vocab,
            vocab_r,
            merges,
            cache,
            dropout: self.config.dropout,
            unk_token: self.config.unk_token,
        })
    }
}

/// A Byte Pair Encoding model.
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
    /// The unknown token to be used when we encounter an unknown char
    unk_token: Option<u32>,
}

impl Default for BPE {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl Clone for BPE {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new BPE will start with a fresh cache.
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            vocab_r: self.vocab_r.clone(),
            merges: self.merges.clone(),
            cache: self.cache.fresh(),
            dropout: self.dropout,
            unk_token: self.unk_token,
        }
    }
}

impl BPE {
    /// Initialize a `BpeBuilder`.
    pub fn builder() -> BpeBuilder {
        BpeBuilder::new()
    }

    /// Create a new BPE model with the given vocab and merges.
    pub fn new(vocab: HashMap<String, u32>, merges: HashMap<Pair, (u32, u32)>) -> Self {
        Self::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap()
    }

    /// Initialize a BPE model from vocab and merges file.
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

        Ok(BPE::new(vocab, merges))
    }

    /// Try resetting the cache. This fails if a lock can't be acquired.
    pub fn try_clear_cache(&self) {
        self.cache.try_clear()
    }

    fn merge_word(&self, w: &str) -> Word {
        let mut word = Word::new();
        for c in w.chars() {
            if let Some(id) = self.vocab.get(&c.to_string()) {
                word.add(*id);
            } else if let Some(unk) = &self.unk_token {
                // Handle UNK token
                word.add(*unk);
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
                                // With probability `dropout` we'll ignore this merge.
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

        word
    }

    fn word_to_tokens(&self, word: &Word, initial_offsets: &(usize, usize)) -> Vec<Token> {
        word.get_chars()
            .iter()
            .zip(word.get_offsets())
            .map(|(id, offsets)| {
                Token::new(
                    *id,
                    self.vocab_r[id].clone(),
                    (initial_offsets.0 + offsets.0, initial_offsets.0 + offsets.1),
                )
            })
            .collect::<Vec<_>>()
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
        let mut cached_words = match self.dropout {
            None => self
                .cache
                .get_values(sentence.iter().map(|(s, _)| s.clone())),
            Some(_) => None, // If using dropout we don't want to use a cached.
        };

        for (i, (w, initial_offsets)) in sentence.iter().enumerate() {
            let tokens = match cached_words {
                None => {
                    // Not using cache since we're using dropout.
                    let word = self.merge_word(&w);
                    self.word_to_tokens(&word, initial_offsets)
                }
                Some(ref mut cache) => {
                    match cache[i].as_ref() {
                        None => {
                            // No cache hit, so re-compute merges.
                            let word = self.merge_word(&w);
                            let tokens = self.word_to_tokens(&word, initial_offsets);
                            // Add to cache.
                            cache[i] = Some(word);
                            tokens
                        }
                        Some(word) => self.word_to_tokens(word, initial_offsets),
                    }
                }
            };

            encoded.extend(tokens);
        }

        // Also update cache
        if let Some(cache) = cached_words {
            let keys_iter = sentence.into_iter().map(|(s, _)| s);
            self.cache.set_values(keys_iter, cache.into_iter());
        }

        Ok(encoded)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn save(&self, folder: &Path, name: &str) -> Result<Vec<PathBuf>> {
        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(&format!("{}-vocab.json", name))]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let serialized = serde_json::to_string(&self.vocab)?;
        vocab_file.write_all(&serialized.as_bytes())?;

        // Write merges.txt
        let merges_path: PathBuf = [folder, Path::new(&format!("{}-merges.txt", name))]
            .iter()
            .collect();
        let mut merges_file = File::create(&merges_path)?;
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        merges_file.write_all(
            &merges
                .into_iter()
                .map(|(pair, _)| {
                    format!("{} {}\n", self.vocab_r[&pair.0], self.vocab_r[&pair.1]).into_bytes()
                })
                .flatten()
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path, merges_path])
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
        let mut bpe = BPE::new(vocab, merges);

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
