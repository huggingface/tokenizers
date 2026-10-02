use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[cfg(feature = "bpe")]
use tk_encode::models::bpe::PipelineBPE;
#[cfg(feature = "unigram")]
use tk_encode::models::unigram::Unigram;
#[cfg(feature = "wordlevel")]
use tk_encode::models::wordlevel::WordLevel;
#[cfg(feature = "wordpiece")]
use tk_encode::models::wordpiece::WordPiece;
use tk_encode::pipeline::{
    NormalizerChain, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
    PipelinePreTokenizer, PipelineToken, PipelineTokenizer, PreTokenizer, PreTokenizerScratch,
    Template, normalize_all,
};
use tk_encode::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};
use tk_encode::{
    DecoderRuntime, PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams,
};

use crate::ProgressFormat;
use crate::error::{Result, TrainingError};
use crate::progress::{ProgressBar, ProgressStyle};
use crate::trainer_builder::TokenizerTrainerBuilder;

#[derive(Debug)]
pub struct TokenizerTrainer<T> {
    pub(crate) params: TrainingParams,
    pub(crate) trainer: T,
    pub(crate) pipeline: TrainingPipeline,
}

#[derive(Debug)]
pub struct TrainingParams {
    pub vocab_size: usize,
    pub special_tokens: Vec<AddedToken>,
    pub unk_token: Option<String>,
    pub progress: ProgressFormat,
    pub byte_level: bool,
}

#[cfg(test)]
impl TrainingParams {
    /// No special tokens, no unknown token, no progress output, not byte-level.
    pub(crate) fn for_tests(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            special_tokens: vec![],
            unk_token: None,
            progress: ProgressFormat::Silent,
            byte_level: false,
        }
    }
}

pub trait ModelTrainer {
    type Model: IntoPipelineModel;

    /// Trains a tokenizer model from the words added through [Self::feed]
    fn train_model(&self, params: &TrainingParams) -> Result<Self::Model>;
    /// Refuses `params` this trainer cannot train with, before any text is read.
    ///
    /// Only a BPE model reads text as bytes, so the default refuses byte-level.
    fn check(&self, params: &TrainingParams) -> Result<()> {
        if params.byte_level {
            return Err(TrainingError::ByteLevelUnsupported);
        }
        Ok(())
    }
    /// Process the iterator of sequence, and adds the resulting words to the ones fed so far
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync;
}

/// A bar for a training step, when `progress` asks for bars.
fn progress_bar(
    progress: ProgressFormat,
    len: u64,
    template: &str,
    message: String,
) -> Option<ProgressBar> {
    (progress == ProgressFormat::Indicatif).then(|| {
        let bar = ProgressBar::new(len);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(template)
                .expect("Invalid progress template"),
        );
        bar.set_message(message);
        bar
    })
}

/// A trained model that becomes one of the [`PipelineModel`] variants a tokenizer runs.
///
/// Fallible because a trained WordPiece still has to build its lookup tables.
pub trait IntoPipelineModel {
    fn into_pipeline_model(self) -> Result<PipelineModel>;
}

impl IntoPipelineModel for PipelineModel {
    fn into_pipeline_model(self) -> Result<PipelineModel> {
        Ok(self)
    }
}

#[cfg(feature = "bpe")]
impl IntoPipelineModel for PipelineBPE {
    fn into_pipeline_model(self) -> Result<PipelineModel> {
        Ok(PipelineModel::BPE(self))
    }
}

#[cfg(feature = "unigram")]
impl IntoPipelineModel for Unigram {
    fn into_pipeline_model(self) -> Result<PipelineModel> {
        Ok(PipelineModel::Unigram(self))
    }
}

#[cfg(feature = "wordlevel")]
impl IntoPipelineModel for WordLevel {
    fn into_pipeline_model(self) -> Result<PipelineModel> {
        Ok(PipelineModel::WordLevel(self))
    }
}

#[cfg(feature = "wordpiece")]
impl IntoPipelineModel for WordPiece {
    fn into_pipeline_model(self) -> Result<PipelineModel> {
        Ok(PipelineModel::WordPiece(self.try_into()?))
    }
}

#[derive(Debug)]
pub(crate) struct TrainingPipeline {
    pub(crate) normalizers: Vec<PipelineNormalizer>,
    pub(crate) pre_tokenizer: PipelinePreTokenizer,
    pub(crate) post_processor: PostProcessorSpec,
    pub(crate) decoder: Option<DecoderRuntime>,
    pub(crate) padding: Option<PaddingSpec>,
    pub(crate) truncation: Option<TruncationParams>,
    pub(crate) role_to_token: BTreeMap<String, String>,
}

impl TrainingPipeline {
    /// The words a trainer learns from: `input` normalized, then pre-tokenized.
    fn process(&self, input: &str) -> Result<Vec<String>> {
        let normalized = normalize_all(&self.normalizers, input, 0)?;
        // A pre-tokenizer's spans hold `u32` offsets.
        if normalized.len() > u32::MAX as usize {
            return Err(TrainingError::SequenceTooLong(normalized.len()));
        }
        let mut spans = vec![];
        self.pre_tokenizer.pre_tokenize(
            &normalized,
            &mut PreTokenizerScratch::default(),
            &mut spans,
        )?;
        Ok(spans
            .into_iter()
            .map(|span| normalized[span.range()].to_owned())
            .collect())
    }

    /// The tokenizer that runs these stages around `model`.
    ///
    /// The special tokens keep the ids the trainer gave them, the first of the vocabulary. The
    /// template tokens and the pad token get the ids of the special tokens they name.
    fn bind(self, model: PipelineModel, params: &TrainingParams) -> Result<PipelineTokenizer> {
        let Self {
            decoder,
            normalizers,
            post_processor,
            pre_tokenizer,
            padding,
            truncation,
            role_to_token,
        } = self;

        let mut added_vocabulary = AddedVocabulary::new();
        // The same normalizers tk-serialize matches normalized added tokens against, so that a
        // trained tokenizer reads back from its `tokenizer.json` unchanged.
        let added_normalizers = match normalizers.last() {
            Some(PipelineNormalizer::Metaspace(_)) => &normalizers[..normalizers.len() - 1],
            _ => &normalizers[..],
        };
        added_vocabulary.add_special_tokens(
            params.special_tokens.iter().cloned(),
            model.id_space(),
            |token| model.token_to_id(token),
            Some(&NormalizerChain(added_normalizers)),
        )?;

        let ids = |run: Vec<(String, u8)>| {
            run.into_iter()
                .map(|(token, type_id)| {
                    let id = added_vocabulary
                        .token_to_id(&token)
                        .ok_or(TrainingError::TemplateTokenNotSpecial(token))?;
                    Ok((PipelineToken::from(id), type_id))
                })
                .collect::<Result<Box<[_]>>>()
        };
        let template = |template: TemplateSpec| -> Result<Template> {
            Ok(Template {
                prefix: ids(template.prefix)?,
                infix: ids(template.infix)?,
                suffix: ids(template.suffix)?,
                a_type_id: template.a_type_id,
                b_type_id: template.b_type_id,
            })
        };
        let post_processor = PipelinePostProcessor {
            single: template(post_processor.single)?,
            pair: template(post_processor.pair)?,
        };
        let padding = padding
            .map(|padding| {
                let pad_id = added_vocabulary
                    .token_to_id(&padding.pad_token)
                    .ok_or_else(|| TrainingError::PadTokenNotSpecial(padding.pad_token.clone()))?;
                Ok::<_, TrainingError>(PaddingParams {
                    strategy: padding.strategy,
                    direction: padding.direction,
                    pad_to_multiple_of: padding.pad_to_multiple_of,
                    pad_id,
                    pad_type_id: padding.pad_type_id,
                    pad_token: padding.pad_token,
                })
            })
            .transpose()?;

        Ok(PipelineTokenizer::from_parts(
            added_vocabulary,
            normalizers,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
            role_to_token,
            padding,
            truncation,
        ))
    }
}

#[derive(Debug)]
pub struct PostProcessorSpec {
    pub single: TemplateSpec,
    pub pair: TemplateSpec,
}

impl Default for PostProcessorSpec {
    /// No special tokens. The second sequence of a pair gets type id 1, like
    /// [`PipelinePostProcessor::default`].
    fn default() -> Self {
        Self {
            single: TemplateSpec::default(),
            pair: TemplateSpec {
                b_type_id: Some(1),
                ..TemplateSpec::default()
            },
        }
    }
}

/// [`PaddingParams`] with the pad token named by content: its id only exists once the model is
/// trained.
#[derive(Debug)]
pub struct PaddingSpec {
    pub strategy: PaddingStrategy,
    pub direction: PaddingDirection,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_type_id: u32,
    pub pad_token: String,
}

#[derive(Debug, Default)]
pub struct TemplateSpec {
    pub prefix: Vec<(String, u8)>,
    pub infix: Vec<(String, u8)>,
    pub suffix: Vec<(String, u8)>,
    pub a_type_id: u8,
    pub b_type_id: Option<u8>,
}

impl<T: ModelTrainer> TokenizerTrainer<T> {
    pub fn builder(trainer: T) -> TokenizerTrainerBuilder<T> {
        TokenizerTrainerBuilder::new(trainer)
    }

    pub fn train<I, S>(self, sequences: I) -> Result<PipelineTokenizer>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let (lower, upper) = sequences.size_hint();
        let progress = progress_bar(
            self.params.progress,
            upper.unwrap_or(lower) as u64,
            "[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}",
            "Pre-processing sequences".to_owned(),
        );
        self.train_counting(sequences, progress, |_| 1)
    }

    /// Trains on `sequences`, advancing `progress` by `size` of each one as it is fed.
    fn train_counting<I, S>(
        self,
        sequences: I,
        progress: Option<ProgressBar>,
        size: impl Fn(&str) -> u64 + Sync,
    ) -> Result<PipelineTokenizer>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let Self {
            params,
            mut trainer,
            pipeline,
        } = self;
        let sequences = sequences.inspect(|sequence| {
            if let Some(progress) = &progress {
                progress.inc(size(sequence.as_ref()));
            }
        });
        trainer.feed(sequences, |text| TrainingPipeline::process(&pipeline, text))?;
        if let Some(progress) = &progress {
            progress.finish();
        }
        let model = trainer.train_model(&params)?.into_pipeline_model()?;
        pipeline.bind(model, &params)
    }

    pub fn train_files<I, P>(self, files: I) -> Result<PipelineTokenizer>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let files = files
            .into_iter()
            .map(File::open)
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut bytes = 0;
        for file in &files {
            bytes += file.metadata()?.len();
        }
        let progress = progress_bar(
            self.params.progress,
            bytes,
            "[{elapsed_precise}] {msg:<30!} {wide_bar} {percent:>18!}%",
            format!("Pre-processing files ({} MB)", bytes / 1_000_000),
        );
        let readers = files.into_iter().map(BufReader::new);

        // Training takes plain lines, so a read error ends the iteration and is returned after
        // training on the lines read before it.
        let mut read_error = None;
        let lines = readers
            .flat_map(|mut reader| {
                std::iter::from_fn(move || {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) => None,
                        Ok(_) => Some(Ok(line)),
                        Err(e) => Some(Err(e)),
                    }
                })
            })
            .map_while(|line| line.map_err(|e| read_error = Some(e)).ok());

        let tokenizer = self.train_counting(lines, progress, |line| line.len() as u64);
        match read_error {
            Some(e) => Err(e.into()),
            None => tokenizer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_encode::normalizers::Lowercase;
    use tk_encode::pre_tokenizers::whitespace::Whitespace;

    fn pipeline(
        normalizers: Vec<PipelineNormalizer>,
        pre_tokenizer: PipelinePreTokenizer,
    ) -> TrainingPipeline {
        TrainingPipeline {
            normalizers,
            pre_tokenizer,
            post_processor: PostProcessorSpec::default(),
            decoder: None,
            padding: None,
            truncation: None,
            role_to_token: BTreeMap::new(),
        }
    }

    #[cfg(feature = "wordlevel")]
    fn padding(pad_token: &str) -> PaddingSpec {
        PaddingSpec {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_to_multiple_of: Some(8),
            pad_type_id: 2,
            pad_token: pad_token.to_owned(),
        }
    }

    #[cfg(feature = "wordlevel")]
    fn truncation() -> TruncationParams {
        TruncationParams {
            max_length: 128,
            ..TruncationParams::default()
        }
    }

    #[test]
    fn process_normalizes_then_pre_tokenizes() {
        let pipeline = pipeline(
            vec![PipelineNormalizer::Lowercase(Lowercase)],
            PipelinePreTokenizer::Whitespace(Whitespace),
        );

        let words = pipeline.process("Hello, World!").unwrap();

        assert_eq!(words, ["hello", ",", "world", "!"]);
    }

    #[test]
    fn process_keeps_the_whole_text_without_a_pre_tokenizer() {
        let pipeline = pipeline(vec![], PipelinePreTokenizer::None);

        let words = pipeline.process("Hello World").unwrap();

        assert_eq!(words, ["Hello World"]);
    }

    #[cfg(feature = "wordlevel")]
    fn wordlevel() -> PipelineModel {
        let vocab = [("[UNK]", 0), ("[CLS]", 1), ("hello", 2)];
        let vocab = vocab.map(|(token, id)| (token.to_owned(), id));
        let model = tk_encode::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        PipelineModel::WordLevel(model)
    }

    #[cfg(feature = "wordlevel")]
    fn params(special_tokens: &[&str]) -> TrainingParams {
        TrainingParams {
            special_tokens: special_tokens
                .iter()
                .map(|token| AddedToken::from(*token, true))
                .collect(),
            ..TrainingParams::for_tests(3)
        }
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn bind_gives_special_tokens_their_model_ids() {
        let pipeline = pipeline(vec![], PipelinePreTokenizer::None);

        let tokenizer = pipeline
            .bind(wordlevel(), &params(&["[UNK]", "[CLS]"]))
            .unwrap();

        let added = tokenizer.get_added_vocabulary();
        assert_eq!(added.token_to_id("[UNK]"), Some(0));
        assert_eq!(added.token_to_id("[CLS]"), Some(1));
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn bind_gives_the_pad_token_its_id() {
        let mut pipeline = pipeline(vec![], PipelinePreTokenizer::None);
        pipeline.padding = Some(padding("[CLS]"));

        let tokenizer = pipeline
            .bind(wordlevel(), &params(&["[UNK]", "[CLS]"]))
            .unwrap();

        let padding = tokenizer.get_padding().unwrap();
        assert_eq!(padding.pad_id, 1);
        assert_eq!(padding.pad_token, "[CLS]");
        assert_eq!(padding.direction, PaddingDirection::Left);
        assert_eq!(padding.pad_to_multiple_of, Some(8));
        assert_eq!(padding.pad_type_id, 2);
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn bind_keeps_truncation_and_roles() {
        let mut pipeline = pipeline(vec![], PipelinePreTokenizer::None);
        pipeline.truncation = Some(truncation());
        pipeline.role_to_token = BTreeMap::from([("cls_token".to_owned(), "[CLS]".to_owned())]);

        let tokenizer = pipeline
            .bind(wordlevel(), &params(&["[UNK]", "[CLS]"]))
            .unwrap();

        assert_eq!(tokenizer.get_truncation(), Some(&truncation()));
        assert_eq!(tokenizer.get_token_for_role("cls_token"), Some("[CLS]"));
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn bind_turns_template_tokens_into_ids() {
        let mut pipeline = pipeline(vec![], PipelinePreTokenizer::None);
        pipeline.post_processor.single.prefix = vec![("[CLS]".to_owned(), 0)];

        let tokenizer = pipeline
            .bind(wordlevel(), &params(&["[UNK]", "[CLS]"]))
            .unwrap();

        let prefix = &tokenizer.get_post_processor().single.prefix;
        let prefix: Vec<_> = prefix
            .iter()
            .map(|(token, type_id)| (token.id(), *type_id))
            .collect();
        assert_eq!(prefix, [(1, 0)]);
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn bind_keeps_the_pipeline_stages() {
        let pipeline = pipeline(
            vec![PipelineNormalizer::Lowercase(Lowercase)],
            PipelinePreTokenizer::Whitespace(Whitespace),
        );

        let tokenizer = pipeline.bind(wordlevel(), &params(&["[UNK]"])).unwrap();

        assert!(matches!(
            tokenizer.get_normalizers(),
            [PipelineNormalizer::Lowercase(_)]
        ));
        assert!(matches!(
            tokenizer.get_pre_tokenizer(),
            PipelinePreTokenizer::Whitespace(_)
        ));
    }

    #[test]
    fn default_pair_template_types_the_second_sequence() {
        assert_eq!(PostProcessorSpec::default().pair.b_type_id, Some(1));
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn bind_gives_a_tokenizer_that_reads_back_from_its_json_unchanged() {
        let mut pipeline = pipeline(
            vec![PipelineNormalizer::Lowercase(Lowercase)],
            PipelinePreTokenizer::Whitespace(Whitespace),
        );
        pipeline.post_processor.single.prefix = vec![("[CLS]".to_owned(), 0)];
        pipeline.padding = Some(padding("[UNK]"));
        pipeline.truncation = Some(truncation());
        pipeline.role_to_token = BTreeMap::from([("cls_token".to_owned(), "[CLS]".to_owned())]);
        let tokenizer = pipeline
            .bind(wordlevel(), &params(&["[UNK]", "[CLS]"]))
            .unwrap();

        let json = tk_serialize::to_json(&tokenizer).unwrap();
        let read_back = tk_serialize::from_json(&json).unwrap();

        assert_eq!(tk_serialize::to_json(&read_back).unwrap(), json);
    }

    fn assert_keeps_vocab(model: impl IntoPipelineModel, token: &str, id: u32) {
        let model = model.into_pipeline_model().unwrap();
        assert_eq!(model.token_to_id(token), Some(id));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn into_pipeline_model_keeps_a_bpe_vocab() {
        let vocab = [("a", 0), ("b", 1), ("ab", 2)].map(|(token, id)| (token.to_owned(), id));
        let model = PipelineBPE::from_config(tk_encode::models::bpe::BpeConfig {
            vocab: vocab.into_iter().collect(),
            merges: vec![("a".to_owned(), "b".to_owned())],
            ..Default::default()
        })
        .unwrap();

        assert_keeps_vocab(model, "ab", 2);
    }

    #[cfg(feature = "unigram")]
    #[test]
    fn into_pipeline_model_keeps_a_unigram_vocab() {
        let pieces = vec![("<unk>".to_owned(), 0.0), ("a".to_owned(), -1.0)];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        assert_keeps_vocab(model, "a", 1);
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn into_pipeline_model_keeps_a_wordlevel_vocab() {
        let vocab = [("[UNK]", 0), ("hello", 1)].map(|(token, id)| (token.to_owned(), id));
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();

        assert_keeps_vocab(model, "hello", 1);
    }

    #[cfg(feature = "wordpiece")]
    #[test]
    fn into_pipeline_model_keeps_a_wordpiece_vocab() {
        let vocab =
            [("[UNK]", 0), ("hell", 1), ("##o", 2)].map(|(token, id)| (token.to_owned(), id));
        let model = WordPiece::builder()
            .vocab(vocab.into_iter().collect::<ahash::AHashMap<_, _>>())
            .build()
            .unwrap();

        assert_keeps_vocab(model, "##o", 2);
    }

    #[cfg(any(feature = "bpe", feature = "wordlevel"))]
    fn ids(tokenizer: &PipelineTokenizer, text: &str) -> Vec<u32> {
        let encodings = tokenizer
            .encode(text, &tk_encode::pipeline::EncodeOptions::default())
            .wait()
            .unwrap();
        encodings[0].ids().iter().map(|token| token.id()).collect()
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn trains_a_tokenizer_that_encodes() {
        let tokenizer = TokenizerTrainer::builder(crate::WordLevelTrainer::default())
            .vocab_size(4)
            .special_tokens(["[UNK]", "[CLS]"])
            .unk_token("[UNK]")
            .pre_tokenizer(PipelinePreTokenizer::Whitespace(Whitespace))
            .post_processor(PostProcessorSpec {
                single: TemplateSpec {
                    prefix: vec![("[CLS]".to_owned(), 0)],
                    ..TemplateSpec::default()
                },
                ..PostProcessorSpec::default()
            })
            .progress(ProgressFormat::Silent)
            .build()
            .unwrap()
            .train(["hello world hello"].iter())
            .unwrap();

        assert_eq!(ids(&tokenizer, "hello world"), [1, 2, 3]);
        assert_eq!(ids(&tokenizer, "hello there"), [1, 2, 0]);
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn trains_a_bpe_tokenizer_that_merges() {
        let tokenizer = TokenizerTrainer::builder(crate::BpeTrainer::default())
            .vocab_size(100)
            .pre_tokenizer(PipelinePreTokenizer::Whitespace(Whitespace))
            .progress(ProgressFormat::Silent)
            .build()
            .unwrap()
            .train(["hello hello hello"].iter())
            .unwrap();

        assert_eq!(ids(&tokenizer, "hello").len(), 1);
    }

    #[cfg(feature = "wordpiece")]
    #[test]
    fn trains_a_wordpiece_tokenizer() {
        let tokenizer = TokenizerTrainer::builder(crate::WordPieceTrainer::builder().build())
            .vocab_size(100)
            .special_tokens(["[UNK]"])
            .unk_token("[UNK]")
            .pre_tokenizer(PipelinePreTokenizer::Whitespace(Whitespace))
            .progress(ProgressFormat::Silent)
            .build()
            .unwrap()
            .train(["hello hello world"].iter())
            .unwrap();

        assert!(!ids(&tokenizer, "hello world").contains(&0));
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn train_files_reads_every_line_of_every_file() {
        let dir = tempfile::tempdir().unwrap();
        let files = [
            ("a.txt", "hello world\nhello"),
            ("b.txt", "hello there\r\nthere"),
        ];
        let files = files.map(|(name, text)| {
            let path = dir.path().join(name);
            std::fs::write(&path, text).unwrap();
            path
        });

        let tokenizer = TokenizerTrainer::builder(crate::WordLevelTrainer::default())
            .vocab_size(4)
            .special_tokens(["[UNK]"])
            .unk_token("[UNK]")
            .pre_tokenizer(PipelinePreTokenizer::Whitespace(Whitespace))
            .progress(ProgressFormat::Indicatif)
            .build()
            .unwrap()
            .train_files(&files)
            .unwrap();

        assert_eq!(ids(&tokenizer, "hello there world"), [1, 2, 3]);
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn trains_a_byte_level_bpe_that_decodes_unseen_characters() {
        let tokenizer = TokenizerTrainer::builder(crate::BpeTrainer::default())
            .vocab_size(300)
            .byte_level()
            .progress(ProgressFormat::Silent)
            .build()
            .unwrap()
            .train(["héllo héllo"].iter())
            .unwrap();

        let text = "héllo wörld ✓";
        let ids = ids(&tokenizer, text);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }
}
