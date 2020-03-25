//! Represents a tokenization pipeline.
//!
//! A [`Tokenizer`](struct.Tokenizer.html) is composed of some of the following parts.
//!   - [`Normalizer`](trait.Normalizer.html): Takes care of the text normalization (like unicode normalization).
//!   - [`PreTokenizer`](trait.PreTokenizer.html): Takes care of the pre tokenization (ie. How to split tokens and pre-process
//!   them.
//!   - [`Model`](trait.Model.html): A model encapsulates the tokenization algorithm (like BPE, Word base, character
//!   based, ...).
//!   - [`PostProcessor`](trait.PostProcessor.html): Takes care of the processing after tokenization (like truncating, padding,
//!   ...).

use crate::utils::iter::ResultShunt;
pub use crate::utils::padding::{pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy};
pub use crate::utils::truncation::{truncate_encodings, TruncationParams, TruncationStrategy};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

mod encoding;
mod normalizer;

pub use encoding::*;
pub use normalizer::*;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type Offsets = (usize, usize);

/// Takes care of pre-processing strings.
pub trait Normalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}

/// The `PreTokenizer` is in charge of doing the pre-segmentation step. It splits the given string
/// in multiple substrings, keeping track of the offsets of said substrings from the
/// `NormalizedString`. In some occasions, the `PreTokenizer` might need to modify the given
/// `NormalizedString` to ensure we can entirely keep track of the offsets and the mapping with
/// the original string.
pub trait PreTokenizer {
    fn pre_tokenize(&self, normalized: &mut NormalizedString) -> Result<Vec<(String, Offsets)>>;
}

/// Represents a model used during Tokenization (like BPE or Word or Unigram).
pub trait Model {
    fn tokenize(&self, tokens: Vec<(String, Offsets)>) -> Result<Vec<Token>>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
    fn get_vocab_size(&self) -> usize;
    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>>;
}

/// A `PostProcessor` has the responsibility to post process an encoded output of the `Tokenizer`.
/// It adds any special tokens that a language model would require.
pub trait PostProcessor {
    /// Returns the number of tokens that will be added during the processing step
    fn added_tokens(&self, is_pair: bool) -> usize;
    /// Process both encodings and returns a new merged one
    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding>;
}
impl dyn PostProcessor {
    pub fn default_process(
        mut encoding: Encoding,
        pair_encoding: Option<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Encoding> {
        match pair_encoding {
            None => Ok(encoding),
            Some(pair) => {
                encoding.merge_with(pair, false);
                Ok(encoding)
            }
        }
    }
}

/// A `Decoder` has the responsibility to merge the given `Vec<String>` in a `String`.
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String>;
}

/// A `Trainer` has the responsibility to train a model. We feed it with lines/sentences
/// and it returns a `Model` when done.
pub trait Trainer: Sync {
    /// Whether we should show progress during the training.
    fn should_show_progress(&self) -> bool;
    /// The actual training method. This will return a new trained Model as well as a list
    /// of `special_tokens` to be added directly to the tokenizer along with the model.
    fn train(
        &self,
        words: HashMap<String, u32>,
    ) -> Result<(Box<dyn Model + Sync>, Vec<AddedToken>)>;
    /// Process a bunch of token, counting them as relevant.
    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>);
}

#[derive(Debug, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
    pub word: u32,
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize), word: u32) -> Self {
        Token {
            id,
            value,
            offsets,
            word,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EncodeInput {
    Single(String),
    Dual(String, String),
}

#[derive(Debug, Clone)]
pub struct AddedToken {
    /// The content of the added token
    pub content: String,
    /// Whether this token must be a single word or can break words
    pub single_word: bool,
    /// Whether this token should strip whitespaces on its left
    pub lstrip: bool,
    /// Whether this token should strip whitespaces on its right
    pub rstrip: bool,
}
impl AddedToken {
    pub fn from(content: String) -> Self {
        AddedToken {
            content,
            ..Default::default()
        }
    }
    pub fn single_word(mut self, single_word: bool) -> Self {
        self.single_word = single_word;
        self
    }
    pub fn lstrip(mut self, lstrip: bool) -> Self {
        self.lstrip = lstrip;
        self
    }
    pub fn rstrip(mut self, rstrip: bool) -> Self {
        self.rstrip = rstrip;
        self
    }
    pub fn get_pattern(&self) -> String {
        let mut r = if self.single_word {
            let first_b = self
                .content
                .chars()
                .next()
                .map(|c| {
                    if regex_syntax::is_word_character(c) {
                        r"\b"
                    } else {
                        ""
                    }
                })
                .unwrap();
            let last_b = self
                .content
                .chars()
                .last()
                .map(|c| {
                    if regex_syntax::is_word_character(c) {
                        r"\b"
                    } else {
                        ""
                    }
                })
                .unwrap();
            format!(r"{}{}{}", first_b, regex::escape(&self.content), last_b)
        } else {
            regex::escape(&self.content)
        };

        if self.lstrip && self.rstrip {
            r = format!(r"(\s)?{}(\s)?", r);
        } else if self.lstrip {
            r = format!(r"(\s)?{}", r);
        } else if self.rstrip {
            r = format!(r"{}(\s)?", r);
        }

        r
    }
}
impl Default for AddedToken {
    fn default() -> Self {
        AddedToken {
            content: String::new(),
            single_word: false,
            lstrip: false,
            rstrip: false,
        }
    }
}
// We only want to hash on the content. AddedToken cannot be added multiple times with different
// options
impl std::hash::Hash for AddedToken {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
    }
}
impl std::cmp::PartialEq for AddedToken {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}
impl std::cmp::Eq for AddedToken {}

/// A `Tokenizer` is capable of encoding/decoding any text.
pub struct Tokenizer {
    // Tokenizer parts
    normalizer: Option<Box<dyn Normalizer + Sync>>,
    pre_tokenizer: Option<Box<dyn PreTokenizer + Sync>>,
    model: Box<dyn Model + Sync>,
    post_processor: Option<Box<dyn PostProcessor + Sync>>,
    decoder: Option<Box<dyn Decoder + Sync>>,

    // Added Vocabulary capabilities
    /// Contains the mapping from String to ID as the user intended it. This map
    /// contains both special tokens and classic added tokens.
    added_tokens_map: HashMap<String, u32>,
    /// Contains the mapping from ID to AddedToken for all the added tokens, both special
    /// and classic.
    added_tokens_map_r: HashMap<u32, AddedToken>,
    /// Contains only the classic AddedToken, in the specific order the user gave them.
    added_tokens: Vec<AddedToken>,
    /// Contains only the special AddedToken, in the specific order the user gave them.
    special_tokens: Vec<AddedToken>,
    /// A Set, containing all the special token for easy access while decoding. This let's
    /// use remove them easily with an O(1) complexity.
    special_tokens_set: HashSet<String>,
    /// A RegexSet containing all the patterns used to split on AddedTokens
    split_re: regex::RegexSet,

    // General processing parameters
    trunc: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl Tokenizer {
    /// Instanciate a new Tokenizer, with the given Model
    pub fn new(model: Box<dyn Model + Sync>) -> Self {
        Tokenizer {
            normalizer: None,
            pre_tokenizer: None,
            model,
            post_processor: None,
            decoder: None,

            added_tokens_map: HashMap::new(),
            added_tokens_map_r: HashMap::new(),
            added_tokens: vec![],
            special_tokens: vec![],
            special_tokens_set: HashSet::new(),
            split_re: regex::RegexSet::new::<_, &&str>(&[]).unwrap(),

            trunc: None,
            padding: None,
        }
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: Box<dyn Normalizer + Sync>) -> &Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Get the normalizer
    #[allow(clippy::borrowed_box)]
    pub fn get_normalizer(&self) -> Option<&Box<dyn Normalizer + Sync>> {
        self.normalizer.as_ref()
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: Box<dyn PreTokenizer + Sync>) -> &Self {
        self.pre_tokenizer = Some(pre_tokenizer);
        self
    }

    /// Get the pre tokenizer
    #[allow(clippy::borrowed_box)]
    pub fn get_pre_tokenizer(&self) -> Option<&Box<dyn PreTokenizer + Sync>> {
        self.pre_tokenizer.as_ref()
    }

    /// Set the post processor
    pub fn with_post_processor(&mut self, post_processor: Box<dyn PostProcessor + Sync>) -> &Self {
        self.post_processor = Some(post_processor);
        self
    }

    /// Get the post processor
    #[allow(clippy::borrowed_box)]
    pub fn get_post_processor(&self) -> Option<&Box<dyn PostProcessor + Sync>> {
        self.post_processor.as_ref()
    }

    /// Set the decoder
    pub fn with_decoder(&mut self, decoder: Box<dyn Decoder + Sync>) -> &Self {
        self.decoder = Some(decoder);
        self
    }

    /// Get the decoder
    #[allow(clippy::borrowed_box)]
    pub fn get_decoder(&self) -> Option<&Box<dyn Decoder + Sync>> {
        self.decoder.as_ref()
    }

    /// Set the model
    pub fn with_model(&mut self, model: Box<dyn Model + Sync>) -> &Self {
        self.model = model;
        self
    }

    /// Get the model
    #[allow(clippy::borrowed_box)]
    pub fn get_model(&self) -> &Box<dyn Model + Sync> {
        &self.model
    }

    /// Set the truncation parameters
    pub fn with_truncation(&mut self, trunc: Option<TruncationParams>) -> &Self {
        self.trunc = trunc;
        self
    }

    /// Set the padding strategy
    pub fn with_padding(&mut self, padding: Option<PaddingParams>) -> &Self {
        self.padding = padding;
        self
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.model.get_vocab_size()
            + if with_added_tokens {
                self.added_tokens.len()
            } else {
                0
            }
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(id) = self.added_tokens_map.get(token) {
            Some(*id)
        } else {
            self.model.token_to_id(token)
        }
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        if let Some(token) = self.added_tokens_map_r.get(&id) {
            Some(token.content.clone())
        } else {
            self.model.id_to_token(id)
        }
    }

    /// Normalize the given sentence and return the corresponding normalized string
    pub fn normalize(&self, sentence: &str) -> Result<NormalizedString> {
        let mut normalized = self
            .split_on_added_tokens(sentence)
            .into_iter()
            .map(|(sentence, id)| -> Result<NormalizedString> {
                if id.is_some() {
                    Ok(sentence)
                } else {
                    let mut normalized = self.do_normalize(sentence)?;
                    let _ = self.pre_tokenize(&mut normalized)?;

                    Ok(normalized)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let others = normalized.split_off(1);
        let mut normalized: NormalizedString = normalized.into_iter().next().unwrap();
        for n in others {
            normalized.merge_with(&n);
        }

        Ok(normalized)
    }

    /// Encode the given sentence
    pub fn encode(&self, input: EncodeInput, add_special_tokens: bool) -> Result<Encoding> {
        let generate_output =
            move |sentence: String, type_id: u32| -> Result<(Encoding, NormalizedString)> {
                // First we need to split into as many sequences as needed to avoid splitting
                // on our added tokens
                let results = self.split_on_added_tokens(&sentence).into_iter().map(
                    |(sentence, id)| -> Result<(Encoding, NormalizedString)> {
                        // If this is one of our added tokens, lets return an encoding directly
                        if let Some(id) = id {
                            return Ok((
                                Encoding::new(
                                    vec![id],
                                    vec![type_id],
                                    vec![sentence.get().to_owned()],
                                    vec![0],
                                    vec![(0, sentence.len())],
                                    vec![0],
                                    vec![1],
                                    vec![],
                                ),
                                sentence,
                            ));
                        }

                        // 1. Normalization
                        let mut normalized = self.do_normalize(sentence)?;

                        // 2. Pre tokenization
                        let pre_tokenized = self.pre_tokenize(&mut normalized)?;

                        // 3. Model
                        let tokens = self.model.tokenize(pre_tokenized)?;
                        let encoding = Encoding::from_tokens(tokens, type_id);

                        Ok((encoding, normalized))
                    },
                );

                let (mut encodings, mut normalized) =
                    ResultShunt::process(results, |iter| iter.unzip::<_, _, Vec<_>, Vec<_>>())?;

                if encodings.is_empty() {
                    return Ok((Encoding::default(), NormalizedString::from("")));
                }

                // Merge encodings and normalized strings
                let others = encodings.split_off(1);
                let n_others = normalized.split_off(1);

                let mut final_encoding: Encoding = encodings.into_iter().next().unwrap();
                let mut final_normalized: NormalizedString = normalized.into_iter().next().unwrap();

                let mut offset = final_normalized.len_original();
                for (mut encoding, normalized) in others.into_iter().zip(n_others) {
                    encoding
                        .get_offsets_mut()
                        .iter_mut()
                        .for_each(|(start, end)| {
                            *start += offset;
                            *end += offset;
                        });
                    offset += normalized.len();

                    final_encoding.merge_with(encoding, false);
                    final_normalized.merge_with(&normalized);
                }

                Ok((final_encoding, final_normalized))
            };

        let (sentence, pair) = match input {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        let (encoding, normalized) = generate_output(sentence, 0)?;
        let (pair_encoding, pair_normalized) = match pair {
            Some(pair) => {
                let (e, n) = generate_output(pair, 1)?;
                (Some(e), Some(n))
            }
            None => (None, None),
        };

        // 4. Post processing
        let mut output = self.post_process(encoding, pair_encoding, add_special_tokens)?;

        // 5. Convert offsets back to original string
        let mut current_offset = (0, 0);
        let mut n_source = &normalized;
        output
            .get_offsets_mut()
            .iter_mut()
            .for_each(|(start, end)| {
                if (*start, *end) < current_offset {
                    n_source = &pair_normalized.as_ref().unwrap_or(&normalized);
                }
                current_offset = (*start, *end);
                let (s, e) = n_source
                    .convert_offsets(Range::Normalized(*start..*end))
                    .map_or((*start, *end), |range| (range.start, range.end));
                *start = s;
                *end = e;
            });

        Ok(output)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch(
        &self,
        inputs: Vec<EncodeInput>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        let encodings = inputs
            .into_par_iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(encodings, &params)
        } else {
            Ok(encodings)
        }
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String> {
        let tokens = ids
            .into_iter()
            .map(|id| {
                let token = if let Some(token) = self.added_tokens_map_r.get(&id) {
                    Some(token.content.to_owned())
                } else {
                    self.model.id_to_token(id)
                };

                token.filter(|token| {
                    !skip_special_tokens || !self.special_tokens_set.contains(token)
                })
            })
            .filter(|token| token.is_some())
            .map(|id| id.unwrap())
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        sentences
            .into_par_iter()
            .map(|sentence| self.decode(sentence, skip_special_tokens))
            .collect()
    }

    /// Train a model and replace our current Model, using the given Trainer
    #[allow(clippy::borrowed_box)]
    pub fn train(&mut self, trainer: &Box<dyn Trainer>, files: Vec<String>) -> Result<()> {
        let progress = ProgressBar::new(100 * files.len() as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>19!}"),
        );
        progress.set_message("Reading files");

        let results = files
            .into_par_iter()
            .map(|filename| -> Result<HashMap<String, u32>> {
                let mut words = HashMap::new();
                let file = File::open(filename)?;
                let len = file.metadata().map_or(0, |c| c.len());
                let mut file = BufReader::new(file);
                let mut prev_prog = 0;
                let mut read = 0;
                let mut curr_prog;

                let mut buf = String::new();
                loop {
                    buf.clear();
                    // We read new lines using this API instead of the Lines Iterator
                    // on purpose. We want to keep the `\n` and potential `\r` between each lines
                    match file.read_line(&mut buf)? {
                        0 => break,
                        b => {
                            let mut normalized = self.do_normalize(NormalizedString::from(&buf))?;
                            let pre_tokenized = self.pre_tokenize(&mut normalized)?;
                            trainer.process_tokens(
                                &mut words,
                                pre_tokenized.into_iter().map(|(t, _)| t).collect(),
                            );

                            read += b as u64;
                            curr_prog = ((read as f64 / len as f64) * 100.0) as u64;
                            if curr_prog > prev_prog {
                                progress.inc(curr_prog - prev_prog);
                                prev_prog = curr_prog;
                            }
                        }
                    }
                }

                Ok(words)
            })
            .collect::<Vec<_>>();
        progress.finish();

        let mut words = HashMap::new();
        for result in results {
            for (word, count) in result? {
                words
                    .entry(word)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
            }
        }

        let (model, special_tokens) = trainer.train(words)?;
        self.model = model;
        self.add_special_tokens(&special_tokens);

        Ok(())
    }

    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    fn pre_tokenize(
        &self,
        mut normalized: &mut NormalizedString,
    ) -> Result<Vec<(String, Offsets)>> {
        match &self.pre_tokenizer {
            None => Ok(vec![(normalized.get().to_owned(), (0, normalized.len()))]),
            Some(pre_tokenizer) => pre_tokenizer.pre_tokenize(&mut normalized),
        }
    }

    /// Normalization logic, go through all normalizers
    fn do_normalize(&self, mut normalized: NormalizedString) -> Result<NormalizedString> {
        if let Some(normalizer) = &self.normalizer {
            normalizer.normalize(&mut normalized)?;
        }

        Ok(normalized)
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    pub fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // 1. First we truncate if needed
        let (encoding, pair_encoding) = {
            if let Some(trunc) = &self.trunc {
                let n_added_tokens = if let Some(processor) = &self.post_processor {
                    processor.added_tokens(pair_encoding.is_some())
                } else {
                    0
                };

                if add_special_tokens && n_added_tokens > 0 {
                    let params = TruncationParams {
                        max_length: trunc.max_length - n_added_tokens,
                        ..*trunc
                    };
                    truncate_encodings(encoding, pair_encoding, &params)?
                } else {
                    truncate_encodings(encoding, pair_encoding, &trunc)?
                }
            } else {
                (encoding, pair_encoding)
            }
        };

        // 2. Then We post process
        let mut final_encoding = if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding, add_special_tokens)?
        } else {
            PostProcessor::default_process(encoding, pair_encoding, add_special_tokens)?
        };

        // 3. Then we pad if needed
        if let Some(params) = &self.padding {
            // We can only pad for a given size. If the Strategy is BatchLongest, it will be done
            // when we handle a batch
            let size = if let PaddingStrategy::Fixed(size) = params.strategy {
                size
            } else {
                final_encoding.get_ids().len()
            };

            final_encoding.pad(
                size,
                params.pad_id,
                params.pad_type_id,
                &params.pad_token,
                params.direction,
            );
        }

        Ok(final_encoding)
    }

    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        for token in tokens {
            if !self.special_tokens_set.contains(&token.content) {
                self.special_tokens.push(token.to_owned());
                self.special_tokens_set.insert(token.content.clone());
            }
        }
        let added = self.add_tokens(&tokens);

        self.refresh_added_tokens();

        added
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        let mut ignored = 0;
        for token in tokens {
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }

            let id = if let Some(id) = self.token_to_id(&token.content) {
                id
            } else {
                let new_id = (self.model.get_vocab_size() + self.added_tokens.len()) as u32;
                self.added_tokens_map.insert(token.content.clone(), new_id);

                if !self.special_tokens_set.contains(&token.content) {
                    self.added_tokens.push(token.clone());
                }

                new_id
            };

            // Update the current revert operation
            self.added_tokens_map_r
                .entry(id)
                .and_modify(|t| *t = token.clone())
                .or_insert_with(|| token.clone());
        }

        self.refresh_added_tokens();

        // Return the number of added tokens
        tokens.len() - ignored
    }

    fn refresh_added_tokens(&mut self) {
        self.split_re = regex::RegexSet::new(
            self.special_tokens
                .iter()
                .chain(self.added_tokens.iter())
                .map(|token| token.get_pattern()),
        )
        .unwrap();
    }

    /// Split the given sentence on multiple parts, finding the added tokens and their id in
    /// the process
    fn split_on_added_tokens(&self, sentence: &str) -> Vec<(NormalizedString, Option<u32>)> {
        let mut matches = self
            .split_re
            .matches(sentence)
            .into_iter()
            .flat_map(|idx| {
                regex::Regex::new(&self.split_re.patterns()[idx])
                    .unwrap()
                    .find_iter(&sentence)
                    .map(|m| (idx, (m.start(), m.end())))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // We sort all the matches by their start and then by their pattern id
        matches.sort_by(
            |(idxa, (sa, _)), (idxb, (sb, _))| {
                if sa != sb {
                    sa.cmp(sb)
                } else {
                    idxa.cmp(idxb)
                }
            },
        );

        // Select the matches (if some are overlapping) we want to keep
        let mut i = 0;
        let mut current_offset = 0;
        let mut splits = Vec::with_capacity(matches.len());
        while i < matches.len() {
            let (idx, (start, end)) = matches[i];

            // current match is before the currentt offset, let's skip it
            if start < current_offset {
                i += 1;
                continue;
            }

            // Find out if we have overlapping neighbors. If so, we keep the one with the lowest
            // idx, and apply it, then continue. All others will be skipped since `current_offset`
            // will have been increased
            if i + 1 < matches.len() {
                if let Some((idx, (s, e))) = matches[i..]
                    .iter()
                    .take_while(|(_, (s, e))| *s < end && start < *e)
                    .min() // Order on idx first
                    .copied()
                {
                    splits.push((idx, (s, e)));
                    current_offset = e;
                    i += 1;
                    continue;
                }
            }

            // We didn't find overlapping neighbors, apply ourself
            splits.push((idx, (start, end)));
            current_offset = end;
            i += 1;
        }

        // We also insert the splits that are inbetween the added tokens, to split the entire string
        let mut start_offset = 0;
        let mut splits = splits
            .into_iter()
            .flat_map(|(idx, (start, end))| {
                let mut splits = vec![];
                if start_offset < start {
                    splits.push((None, (start_offset, start)));
                }
                splits.push((Some(idx), (start, end)));
                start_offset = end;

                splits
            })
            .collect::<Vec<_>>();
        if let Some((_, (_, end))) = splits.iter().last().copied() {
            if end < sentence.len() {
                splits.push((None, (end, sentence.len())));
            }
        }

        if splits.is_empty() {
            vec![(NormalizedString::from(sentence), None)]
        } else {
            splits
                .into_iter()
                .map(|(idx, (start, end))| unsafe {
                    let s = sentence.get_unchecked(start..end).to_owned();
                    let mut normalized = NormalizedString::from(&s);

                    // Find out the associated AddedToken, and its id
                    let id = if let Some(idx) = idx {
                        let added = if idx >= self.special_tokens.len() {
                            &self.added_tokens[idx - self.special_tokens.len()]
                        } else {
                            &self.special_tokens[idx]
                        };

                        if added.lstrip && added.rstrip {
                            normalized.strip();
                        } else if added.lstrip {
                            normalized.lstrip();
                        } else if added.rstrip {
                            normalized.rstrip();
                        }

                        self.token_to_id(&added.content)
                    } else {
                        None
                    };

                    (normalized, id)
                })
                .collect()
        }
    }
}
