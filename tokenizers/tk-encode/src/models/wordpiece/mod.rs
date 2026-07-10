//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::added_vocabulary::bucket_vocab_store::BucketVocabStore;
use crate::models::bpe::BPE;
use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::{Model, Result, Token};
use ahash::AHashMap;
use std::collections::HashMap;
use std::{
    borrow::Cow,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

mod serialization;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("WordPiece error: Missing [UNK] token from the vocabulary")]
    MissingUnkToken,
}

type Vocab = AHashMap<String, u32>;
type VocabR = AHashMap<u32, String>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

/// A `WordPieceBuilder` can be used to create a `WordPiece` model with a custom configuration.
pub struct WordPieceBuilder {
    config: Config,
}

impl Default for WordPieceBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: AHashMap::new(),
                unk_token: String::from("[UNK]"),
                continuing_subword_prefix: String::from("##"),
                max_input_chars_per_word: 100,
            },
        }
    }
}

impl WordPieceBuilder {
    /// Construct a new `WordPieceBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    #[must_use]
    pub fn vocab<V: Into<AHashMap<String, u32>>>(mut self, vocab: V) -> Self {
        self.config.vocab = vocab.into();
        self
    }

    /// The the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Set the prefix for continuing subwords.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    #[must_use]
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    /// Constructs a `WordPiece` model that uses the `WordPieceBuilder`'s configuration.
    pub fn build(mut self) -> Result<WordPiece> {
        if let Some(vocab) = self.config.files {
            self.config.vocab = WordPiece::read_file(&vocab)?;
        }

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        Ok(WordPiece {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

/// A
/// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
#[derive(Clone, PartialEq, Eq)]
pub struct WordPiece {
    pub vocab: Vocab,
    pub vocab_r: VocabR,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl std::fmt::Debug for WordPiece {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("WordPiece")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            vocab: AHashMap::new(),
            vocab_r: AHashMap::new(),
            unk_token: String::from("[UNK]"),
            continuing_subword_prefix: String::from("##"),
            max_input_chars_per_word: 100,
        }
    }
}

impl WordPiece {
    /// Get a `WordPieceBuilder`.
    pub fn builder() -> WordPieceBuilder {
        WordPieceBuilder::new()
    }

    /// Read the given files to extract the vocab
    pub fn read_file(vocab: &str) -> Result<Vocab> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = AHashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    pub fn read_bytes(vocab: &[u8]) -> Result<Vocab> {
        let file = BufReader::new(vocab);

        let mut vocab = AHashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self> {
        let tokenizer = serde_json::from_slice(bytes.as_ref())?;
        Ok(tokenizer)
    }

    /// Initialize a `WordPiece` model from a vocab mapping file.
    pub fn from_file(vocab: &str) -> WordPieceBuilder {
        WordPiece::builder().files(vocab.to_owned())
    }

    /// Create a `WordPiece` model from a `BPE` model.
    pub fn from_bpe(bpe: &BPE) -> Self {
        let mut wp = Self::builder()
            .vocab(bpe.get_vocab().into_iter().collect::<AHashMap<_, _>>())
            .build()
            .unwrap();
        if let Some(unk) = bpe.get_unk_token() {
            unk.clone_into(&mut wp.unk_token);
        }
        if let Some(prefix) = bpe.get_continuing_subword_prefix() {
            prefix.clone_into(&mut wp.continuing_subword_prefix);
        }
        wp
    }
}

impl Model for WordPiece {
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }]);
        }

        let mut is_bad = false;
        let mut start = 0;
        let mut sub_tokens: Vec<Token> = vec![];

        while start < sequence.len() {
            let mut end = sequence.len();
            let mut cur_str = None;

            while start < end {
                let mut substr: Cow<str> = Cow::Borrowed(&sequence[start..end]);

                if start > 0 {
                    substr = Cow::Owned(format!("{}{}", self.continuing_subword_prefix, substr));
                }
                if self.vocab.contains_key(substr.as_ref()) {
                    cur_str = Some(Token {
                        id: self.vocab[substr.as_ref()],
                        value: substr.to_string(),
                        offsets: (start, end),
                    });
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }

            if cur_str.is_none() {
                is_bad = true;
                break;
            }

            sub_tokens.push(cur_str.unwrap());
            start = end;
        }

        if is_bad {
            Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }])
        } else {
            Ok(sub_tokens)
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.txt"),
            None => "vocab.txt".to_string(),
        };

        // Write vocab.txt
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let mut vocab: Vec<(&String, &u32)> = self.vocab.iter().collect();
        vocab.sort_unstable_by_key(|k| *k.1);
        vocab_file.write_all(
            &vocab
                .into_iter()
                .flat_map(|(token, _)| format!("{token}\n").as_bytes().to_owned())
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }
}

impl pipeline::Model for WordPiece {
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        output: &mut Vec<pipeline::PipelineToken>,
    ) -> Result<()> {
        let mut candidate = String::with_capacity(self.max_input_chars_per_word);
        let mut candidate_tokens = Vec::with_capacity(sequence.len());

        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            let unk_id = *self
                .vocab
                .get(&self.unk_token)
                .ok_or(Error::MissingUnkToken)?;
            output.push(PipelineToken { id: unk_id });
            return Ok(());
        }

        let mut start = 0;

        while start < sequence.len() {
            candidate.clear();
            if start > 0 {
                candidate.push_str(&self.continuing_subword_prefix);
            }
            candidate.push_str(&sequence[start..]);

            let prefix_len = candidate.len() - (sequence.len() - start);
            let matched = sequence[start..]
                .char_indices()
                .rev()
                .find_map(|(idx, character)| {
                    let end = start + idx + character.len_utf8();
                    candidate.truncate(prefix_len + (end - start));
                    self.vocab.get(&candidate).map(|&id| (end, id))
                });
            let Some((end, token_id)) = matched else {
                let unk_id = *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?;
                output.push(PipelineToken { id: unk_id });
                return Ok(());
            };
            candidate_tokens.push(PipelineToken { id: token_id });
            start = end;
        }
        output.extend_from_slice(&candidate_tokens);
        Ok(())
    }
}

/// Pipeline WordPiece: the `Model` fast path backed by the MPHF [`BucketVocabStore`] with a streamed,
/// prefetched longest-prefix match instead of the legacy dependent probe loop. Mirrors `PipelineBPE`.
#[derive(Clone)]
pub struct PipelineWordPiece {
    vocab: BucketVocabStore,
    /// Resolved lazily — an empty/`from_bpe` model may legitimately lack `[UNK]` until a word misses.
    unk_id: Option<u32>,
    /// `##` (the configured continuing-subword prefix) as bytes; prepended to non-initial candidates.
    continuing_subword_prefix: Vec<u8>,
    max_input_chars_per_word: usize,
    /// Distinct token byte-lengths in the vocab, **descending**. At each position we probe only these
    /// (longest first, stop at the first hit) instead of shrinking one char at a time — a length no token
    /// has can't match, so it's never probed; the descending order gives the greedy longest match.
    ///
    /// NOTE: I also tried bucketing these by the candidate's first byte (probe only lengths of tokens
    /// starting with `word[start]`). On a real bert vocab that was *slower*: WordPiece finds a match in a
    /// probe or two, so the fewer-lengths win is tiny, while the 256-way `Vec<Vec>` indirection adds a
    /// cache miss per position. The flat list wins on real data.
    lens_desc: Box<[u16]>,
}

impl PipelineWordPiece {
    pub fn from_wordpiece(wp: &WordPiece) -> Self {
        let tokens: Vec<(Vec<u8>, u32)> = wp
            .vocab
            .iter()
            .map(|(s, &id)| (s.as_bytes().to_vec(), id))
            .collect();
        let mut lens_desc: Vec<u16> = tokens.iter().map(|(s, _)| s.len() as u16).collect();
        lens_desc.sort_unstable();
        lens_desc.dedup();
        lens_desc.reverse();
        // `BucketVocabStore::build` panics on an empty vocab (max_id over no tokens); `new` is the empty ctor.
        let vocab = if tokens.is_empty() {
            BucketVocabStore::new()
        } else {
            BucketVocabStore::build(tokens)
        };
        Self {
            unk_id: wp.vocab.get(&wp.unk_token).copied(),
            vocab,
            continuing_subword_prefix: wp.continuing_subword_prefix.as_bytes().to_vec(),
            max_input_chars_per_word: wp.max_input_chars_per_word,
            lens_desc: lens_desc.into_boxed_slice(),
        }
    }
}

thread_local! {
    /// One reused candidate buffer per thread — the WordPiece hot path is called once per pre-token, so a
    /// fresh `Vec` here was a heap alloc *per word* (measured ~half the model-lookup cost). Reused, it's
    /// zero-alloc after warmup; per-thread so parallel encode stays lock-free.
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl pipeline::Model for PipelineWordPiece {
    fn tokenize_pipeline(&self, sequence: &str, output: &mut Vec<PipelineToken>) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }
        // `chars` <= `bytes`, so only pay the O(len) char count when the byte length could exceed the
        // cap — for a normal word (< max bytes) the decode is skipped entirely.
        if sequence.len() > self.max_input_chars_per_word
            && sequence.chars().count() > self.max_input_chars_per_word
        {
            output.push(PipelineToken {
                id: self.unk_id.ok_or(Error::MissingUnkToken)?,
            });
            return Ok(());
        }

        let prefix = &self.continuing_subword_prefix;
        let bytes = sequence.as_bytes();
        let max_token_len = self.lens_desc.first().copied().unwrap_or(0) as usize;
        let start_out = output.len();

        SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            // The prefix is constant across the word: write it into the scratch once, then each
            // continuation only rewrites the word part (`truncate` back to the prefix, append word).
            buf.clear();
            buf.extend_from_slice(prefix);
            let mut start = 0usize;
            while start < sequence.len() {
                let prefix_len = if start > 0 { prefix.len() } else { 0 };
                let remaining = sequence.len() - start;
                // A match is at most `max_token_len` bytes; cap the word part there and by what's left.
                let cap_word = max_token_len.saturating_sub(prefix_len).min(remaining);

                // Candidate base bytes. Word-initial (`start == 0`, no prefix) probes the word slice
                // directly — no copy; continuations reuse the prefix already in `buf`, appending the word.
                let cand: &[u8] = if prefix_len == 0 {
                    &bytes[start..start + cap_word]
                } else {
                    buf.truncate(prefix.len());
                    buf.extend_from_slice(&bytes[start..start + cap_word]);
                    &buf
                };

                // Probe real token lengths longest-first; first hit is the greedy answer. Byte-exact: a
                // length that splits a codepoint or that no token has can't match a valid-UTF-8 token.
                let mut matched = None;
                for &l in self.lens_desc.iter() {
                    let l = l as usize;
                    if l < prefix_len + 1 {
                        break;
                    }
                    let word_bytes = l - prefix_len;
                    if word_bytes > cap_word {
                        continue;
                    }
                    if let Some(id) = self.vocab.get_bytes(&cand[..prefix_len + word_bytes]) {
                        matched = Some((word_bytes, id));
                        break;
                    }
                }

                match matched {
                    Some((word_bytes, id)) => {
                        output.push(PipelineToken { id });
                        start += word_bytes;
                    }
                    None => {
                        // Whole word is unknown: drop partial sub-tokens, emit one unk (legacy parity).
                        output.truncate(start_out);
                        output.push(PipelineToken {
                            id: self.unk_id.ok_or(Error::MissingUnkToken)?,
                        });
                        return Ok(());
                    }
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Model as _;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }

    /// A small `##` vocab exercising: whole-word hit, greedy split, single-char fallback, unk, and
    /// **multibyte** tokens (é = 2B, 中 = 3B) — the length-set path probes byte lengths that can split a
    /// codepoint, so a splitting length must never produce a false match.
    fn toy() -> WordPiece {
        let pairs = [
            ("[UNK]", 0),
            ("un", 1),
            ("##want", 2),
            ("##ed", 3),
            ("un", 1),
            ("runn", 4),
            ("##ing", 5),
            ("a", 6),
            ("##b", 7),
            ("##c", 8),
            ("play", 9),
            ("caf", 10),
            ("##é", 11),
            ("中", 12),
            ("##中", 13),
            ("中文", 14),
        ];
        let vocab: AHashMap<String, u32> = pairs.iter().map(|(s, i)| (s.to_string(), *i)).collect();
        WordPiece::builder().vocab(vocab).build().unwrap()
    }

    /// `PipelineWordPiece` (length-set MPHF probe) must be byte-exact with the legacy AHashMap path.
    #[test]
    fn pipeline_wordpiece_matches_legacy() {
        let wp = toy();
        let fast = PipelineWordPiece::from_wordpiece(&wp);
        for word in [
            "unwanted",
            "running",
            "unwanteded",
            "abc",
            "play",
            "unk",
            "xyz",
            "a",
            "",
            "café",
            "中文中",
            "中b",
            "caféunwant",
        ] {
            let mut legacy = Vec::new();
            wp.tokenize_pipeline(word, &mut legacy).unwrap();
            let mut streamed = Vec::new();
            fast.tokenize_pipeline(word, &mut streamed).unwrap();
            let legacy: Vec<u32> = legacy.iter().map(|t| t.id).collect();
            let streamed: Vec<u32> = streamed.iter().map(|t| t.id).collect();
            assert_eq!(streamed, legacy, "diverged on {word:?}");
        }
    }
}
