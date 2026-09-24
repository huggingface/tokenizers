//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::Result;
use crate::utils::DEFAULT_CACHE_CAPACITY;
use crate::utils::word_cache::{Lookup, WordCache};
use ahash::AHashMap;
use yada::DoubleArray;
use yada::builder::DoubleArrayBuilder;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("WordPiece error: Missing [UNK] token from the vocabulary")]
    MissingUnkToken,
}

type Vocab = AHashMap<String, u32>;

pub struct WordPieceConfig {
    pub vocab: Vocab,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl std::fmt::Debug for WordPieceConfig {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("WordPieceConfig")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for WordPieceConfig {
    fn default() -> Self {
        Self {
            vocab: AHashMap::new(),
            unk_token: String::from("[UNK]"),
            continuing_subword_prefix: String::from("##"),
            max_input_chars_per_word: 100,
        }
    }
}

pub struct WordPieceScratch {
    candidate_str: String,
    /// Outlives the encode call that fills it, or it would never see a word twice.
    word_cache: WordCache,
}

impl pipeline::ModelScratch for WordPieceScratch {}

pub struct WordPiece {
    vocab_trie: yada::DoubleArray<Vec<u8>>,
    vocab_r: Box<[Option<Box<str>>]>,
    unk_token: Option<u32>,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl WordPiece {
    pub fn from_config(config: WordPieceConfig) -> Result<Self> {
        let WordPieceConfig {
            vocab,
            unk_token,
            continuing_subword_prefix,
            max_input_chars_per_word,
        } = config;
        let unk_token = vocab.get(&unk_token).copied();

        // yada requires the keyset sorted by key bytes.
        let mut keyset: Vec<_> = vocab.into_iter().collect();
        keyset.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        let vocab_trie = DoubleArray::new(DoubleArrayBuilder::build(&keyset)?)?;
        let max_id = keyset.iter().map(|&(_, id)| id).max().unwrap_or(0) as usize;
        let mut vocab_r = vec![None; max_id + 1];
        for (token, id) in keyset {
            vocab_r[id as usize] = Some(token.into_boxed_str());
        }

        Ok(Self {
            continuing_subword_prefix,
            max_input_chars_per_word,
            unk_token,
            vocab_trie,
            vocab_r: vocab_r.into_boxed_slice(),
        })
    }

    /// One word, greedily: the longest vocabulary entry it starts with, then the
    /// longest entry the rest of it starts with once the continuing-subword prefix
    /// is put in front, and so on. A piece with no entry at all anywhere in the
    /// word makes the whole word one unk token.
    fn tokenize_word(
        &self,
        sequence: &str,
        candidate: &mut String,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let checkpoint = output.len();

        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            let unk_id = self.unk_token.ok_or(Error::MissingUnkToken)?;
            output.push(PipelineToken::from(unk_id));
            return Ok(());
        }

        let mut start = 0;

        while start < sequence.len() {
            candidate.clear();
            let prefix_len = if start > 0 {
                candidate.push_str(&self.continuing_subword_prefix);
                self.continuing_subword_prefix.len()
            } else {
                0
            };
            candidate.push_str(&sequence[start..]);

            // Matches must extend past the continuing-subword prefix: the
            // prefix alone (or a fragment of it) is not a valid subword here,
            // even if it happens to be in the vocab.
            let Some((token_id, match_len)) = self
                .vocab_trie
                .common_prefix_search(&candidate)
                .filter(|(_, len)| *len > prefix_len)
                .last()
            else {
                let unk_id = self.unk_token.ok_or(Error::MissingUnkToken)?;
                output.truncate(checkpoint);
                output.push(PipelineToken::from(unk_id));
                return Ok(());
            };
            output.push(PipelineToken::from(token_id));
            start += match_len - prefix_len;
        }
        Ok(())
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(id as usize)?.as_deref().map(str::to_owned)
    }

    /// `{"token": id}`, in id order. For a writer; the reverse table is dense over the ids, so the
    /// holes a config left are the `None`s that get skipped.
    pub fn vocab(&self) -> Vec<(String, u32)> {
        self.vocab_r
            .iter()
            .enumerate()
            .filter_map(|(id, token)| Some((token.as_deref()?.to_string(), id as u32)))
            .collect()
    }

    /// The unknown token, or `None` when the config named one that is not in the vocabulary -- the
    /// lowering keeps the id, so a name it could not resolve is not recoverable.
    pub fn unk_token(&self) -> Option<&str> {
        self.vocab_r.get(self.unk_token? as usize)?.as_deref()
    }

    pub fn continuing_subword_prefix(&self) -> &str {
        &self.continuing_subword_prefix
    }

    pub fn max_input_chars_per_word(&self) -> usize {
        self.max_input_chars_per_word
    }
}

impl pipeline::Model for WordPiece {
    type Scratch = WordPieceScratch;

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            candidate_str: String::with_capacity(self.max_input_chars_per_word),
            word_cache: WordCache::new(DEFAULT_CACHE_CAPACITY),
        }
    }

    /// A hit skips `tokenize_word`: one trie search per piece of the word,
    /// each over a fresh copy of what is left to match.
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<pipeline::PipelineToken>,
    ) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }
        let WordPieceScratch {
            candidate_str,
            word_cache,
        } = scratch;

        let placement = match word_cache.lookup(sequence.as_bytes()) {
            Lookup::Hit(ids) => {
                output.extend(ids.iter().copied().map(PipelineToken::from));
                return Ok(());
            }
            Lookup::Miss(at) => at,
        };

        let start = output.len();
        self.tokenize_word(sequence, candidate_str, output)?;

        word_cache.insert(placement, output[start..].iter().map(|token| token.id()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }

    /// `hello` is in the vocabulary whole and as `hell` + `##o`, so the
    /// longest-match walk has something to choose; `world` gives a second
    /// one-token word.
    fn pipeline_wordpiece() -> WordPiece {
        let vocab: Vocab = [
            ("[UNK]", 0u32),
            ("hell", 1),
            ("##o", 2),
            ("hello", 3),
            ("world", 4),
        ]
        .into_iter()
        .map(|(token, id)| (token.to_string(), id))
        .collect();
        WordPiece::from_config(WordPieceConfig {
            vocab,
            max_input_chars_per_word: 8,
            ..Default::default()
        })
        .unwrap()
    }

    fn pipeline_ids(
        model: &WordPiece,
        sequence: &str,
        scratch: &mut WordPieceScratch,
    ) -> Vec<u32> {
        let mut output = vec![];
        pipeline::Model::tokenize_pipeline(model, sequence, scratch, &mut output).unwrap();
        output.iter().map(|token| token.id()).collect()
    }

    #[test]
    fn pipeline_remembers_what_a_word_encoded_to() {
        let model = pipeline_wordpiece();
        let mut scratch = pipeline::Model::init_scratch(&model);

        let ids = pipeline_ids(&model, "hello", &mut scratch);

        assert_eq!(scratch.word_cache.lookup(b"hello").hit(), Some(&ids[..]));
    }

    #[test]
    fn cache_hits_agree_with_a_cold_run() {
        let model = pipeline_wordpiece();
        let long = "hello".repeat(300);
        let corpus = [
            "hello",
            // No id for `##w`, so the whole word is one unk token.
            "hellow",
            // Both again, so these two are served from the cache.
            "hello",
            "hellow",
            // Past `max_input_chars_per_word`, which is another unk token.
            "hellohello",
            "hellohello",
            // Out of the vocabulary, and multibyte.
            "東京",
            "東京",
            // 1500 bytes, past the longest word the cache will store.
            long.as_str(),
            long.as_str(),
        ];

        let mut warm_scratch = pipeline::Model::init_scratch(&model);
        let warm = corpus.map(|sequence| pipeline_ids(&model, sequence, &mut warm_scratch));
        let cold = corpus.map(|sequence| {
            let mut scratch = pipeline::Model::init_scratch(&model);
            pipeline_ids(&model, sequence, &mut scratch)
        });

        assert_eq!(warm, cold);
    }

    #[test]
    fn caches_only_the_ids_this_word_produced() {
        // Every word the pipeline hands the model appends to one output buffer,
        // so a word has to remember its own ids, not everything the buffer holds.
        let model = pipeline_wordpiece();
        let mut scratch = pipeline::Model::init_scratch(&model);
        let mut output = vec![];
        pipeline::Model::tokenize_pipeline(&model, "hello", &mut scratch, &mut output).unwrap();
        pipeline::Model::tokenize_pipeline(&model, "world", &mut scratch, &mut output).unwrap();

        let ids: Vec<u32> = output.iter().map(|token| token.id()).collect();
        assert_eq!(ids, [3, 4]);
        assert_eq!(scratch.word_cache.lookup(b"world").hit(), Some(&[4u32][..]));
    }
}
