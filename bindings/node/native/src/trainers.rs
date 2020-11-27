extern crate tokenizers as tk;

use crate::extraction::*;
use crate::models::Model;
use crate::tokenizer::AddedToken;
use neon::prelude::*;
use std::sync::{Arc, RwLock};

use tk::models::{
    bpe::BpeTrainer, unigram::UnigramTrainer, wordlevel::WordLevelTrainer,
    wordpiece::WordPieceTrainer, TrainerWrapper,
};

/// Trainer
#[derive(Clone)]
pub struct Trainer {
    pub trainer: Option<Arc<RwLock<TrainerWrapper>>>,
}

impl From<TrainerWrapper> for Trainer {
    fn from(trainer: TrainerWrapper) -> Self {
        Self {
            trainer: Some(Arc::new(RwLock::new(trainer))),
        }
    }
}

impl tk::Trainer for Trainer {
    type Model = Model;

    fn should_show_progress(&self) -> bool {
        self.trainer
            .as_ref()
            .expect("Uninitialized Trainer")
            .read()
            .unwrap()
            .should_show_progress()
    }

    fn train(&self, model: &mut Self::Model) -> tk::Result<Vec<tk::AddedToken>> {
        let special_tokens = self
            .trainer
            .as_ref()
            .ok_or("Uninitialized Trainer")?
            .read()
            .unwrap()
            .train(
                &mut model
                    .model
                    .as_ref()
                    .ok_or("Uninitialized Model")?
                    .write()
                    .unwrap(),
            )?;

        Ok(special_tokens)
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> tk::Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> tk::Result<Vec<String>> + Sync,
    {
        self.trainer
            .as_ref()
            .ok_or("Uninitialized Trainer")?
            .write()
            .unwrap()
            .feed(iterator, process)
    }
}

declare_types! {
    pub class JsTrainer for Trainer {
        init(_) {
            // This should not be called from JS
            Ok(Trainer { trainer: None })
        }
    }
}

// BPE

struct BpeTrainerOptions(BpeTrainer);
impl From<BpeTrainerOptions> for BpeTrainer {
    fn from(v: BpeTrainerOptions) -> Self {
        v.0
    }
}
impl FromJsValue for BpeTrainerOptions {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        if let Ok(options) = from.downcast::<JsObject>() {
            let mut builder = BpeTrainer::builder();

            if let Ok(size) = options.get(cx, "vocabSize") {
                if let Some(size) = Option::from_value(size, cx)? {
                    builder = builder.vocab_size(size);
                }
            }
            if let Ok(freq) = options.get(cx, "minFrequency") {
                if let Some(freq) = Option::from_value(freq, cx)? {
                    builder = builder.min_frequency(freq);
                }
            }
            if let Ok(tokens) = options.get(cx, "specialTokens") {
                if tokens.downcast::<JsNull>().is_err() && tokens.downcast::<JsUndefined>().is_err()
                {
                    builder = builder.special_tokens(
                        tokens
                            .downcast::<JsArray>()
                            .map_err(|e| Error(format!("{}", e)))?
                            .to_vec(cx)?
                            .into_iter()
                            .map(|token| Ok(AddedToken::from_value(token, cx)?.into()))
                            .collect::<Result<Vec<_>, Error>>()?,
                    );
                }
            }
            if let Ok(limit) = options.get(cx, "limitAlphabet") {
                if let Some(limit) = Option::from_value(limit, cx)? {
                    builder = builder.limit_alphabet(limit);
                }
            }
            if let Ok(alphabet) = options.get(cx, "initialAlphabet") {
                if let Some(alphabet) = Option::from_value(alphabet, cx)? {
                    builder = builder.initial_alphabet(alphabet);
                }
            }
            if let Ok(show) = options.get(cx, "showProgress") {
                if let Some(show) = Option::from_value(show, cx)? {
                    builder = builder.show_progress(show);
                }
            }
            if let Ok(prefix) = options.get(cx, "continuingSubwordPrefix") {
                if let Some(prefix) = Option::from_value(prefix, cx)? {
                    builder = builder.continuing_subword_prefix(prefix);
                }
            }
            if let Ok(suffix) = options.get(cx, "endOfWordSuffix") {
                if let Some(suffix) = Option::from_value(suffix, cx)? {
                    builder = builder.end_of_word_suffix(suffix);
                }
            }

            Ok(Self(builder.build()))
        } else {
            Err(Error("Expected options type: object".into()))
        }
    }
}

/// bpe_trainer(options?: {
///   vocabSize?: number = 30000,
///   minFrequency?: number = 2,
///   specialTokens?: (string | AddedToken)[] = [],
///   limitAlphabet?: number = undefined,
///   initialAlphabet?: string[] = [],
///   showProgress?: bool = true,
///   continuingSubwordPrefix?: string = undefined,
///   endOfWordSuffix?: string = undefined,
/// })
fn bpe_trainer(mut cx: FunctionContext) -> JsResult<JsTrainer> {
    let trainer = cx
        .extract_opt::<BpeTrainerOptions>(0)?
        .map_or_else(|| BpeTrainer::builder().build(), |o| o.into());

    let mut js_trainer = JsTrainer::new::<_, JsTrainer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_trainer.borrow_mut(&guard).trainer = Some(Arc::new(RwLock::new(trainer.into())));

    Ok(js_trainer)
}

// WordPiece

struct WordPieceTrainerOptions(WordPieceTrainer);
impl From<WordPieceTrainerOptions> for WordPieceTrainer {
    fn from(v: WordPieceTrainerOptions) -> Self {
        v.0
    }
}
impl FromJsValue for WordPieceTrainerOptions {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        if let Ok(options) = from.downcast::<JsObject>() {
            let mut builder = WordPieceTrainer::builder();

            if let Ok(size) = options.get(cx, "vocabSize") {
                if let Some(size) = Option::from_value(size, cx)? {
                    builder = builder.vocab_size(size);
                }
            }
            if let Ok(freq) = options.get(cx, "minFrequency") {
                if let Some(freq) = Option::from_value(freq, cx)? {
                    builder = builder.min_frequency(freq);
                }
            }
            if let Ok(tokens) = options.get(cx, "specialTokens") {
                if tokens.downcast::<JsNull>().is_err() && tokens.downcast::<JsUndefined>().is_err()
                {
                    builder = builder.special_tokens(
                        tokens
                            .downcast::<JsArray>()
                            .map_err(|e| Error(format!("{}", e)))?
                            .to_vec(cx)?
                            .into_iter()
                            .map(|token| Ok(AddedToken::from_value(token, cx)?.into()))
                            .collect::<Result<Vec<_>, Error>>()?,
                    );
                }
            }
            if let Ok(limit) = options.get(cx, "limitAlphabet") {
                if let Some(limit) = Option::from_value(limit, cx)? {
                    builder = builder.limit_alphabet(limit);
                }
            }
            if let Ok(alphabet) = options.get(cx, "initialAlphabet") {
                if let Some(alphabet) = Option::from_value(alphabet, cx)? {
                    builder = builder.initial_alphabet(alphabet);
                }
            }
            if let Ok(show) = options.get(cx, "showProgress") {
                if let Some(show) = Option::from_value(show, cx)? {
                    builder = builder.show_progress(show);
                }
            }
            if let Ok(prefix) = options.get(cx, "continuingSubwordPrefix") {
                if let Some(prefix) = Option::from_value(prefix, cx)? {
                    builder = builder.continuing_subword_prefix(prefix);
                }
            }
            if let Ok(suffix) = options.get(cx, "endOfWordSuffix") {
                if let Some(suffix) = Option::from_value(suffix, cx)? {
                    builder = builder.end_of_word_suffix(suffix);
                }
            }

            Ok(Self(builder.build()))
        } else {
            Err(Error("Expected options type: object".into()))
        }
    }
}

/// wordpiece_trainer(options?: {
///   vocabSize?: number = 30000,
///   minFrequency?: number = 2,
///   specialTokens?: string[] = [],
///   limitAlphabet?: number = undefined,
///   initialAlphabet?: string[] = [],
///   showProgress?: bool = true,
///   continuingSubwordPrefix?: string = undefined,
///   endOfWordSuffix?: string = undefined,
/// })
fn wordpiece_trainer(mut cx: FunctionContext) -> JsResult<JsTrainer> {
    let trainer = cx
        .extract_opt::<WordPieceTrainerOptions>(0)?
        .map_or_else(|| WordPieceTrainer::builder().build(), |o| o.into());

    let mut js_trainer = JsTrainer::new::<_, JsTrainer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_trainer.borrow_mut(&guard).trainer = Some(Arc::new(RwLock::new(trainer.into())));

    Ok(js_trainer)
}

// WordLevel

struct WordLevelTrainerOptions(WordLevelTrainer);
impl From<WordLevelTrainerOptions> for WordLevelTrainer {
    fn from(v: WordLevelTrainerOptions) -> Self {
        v.0
    }
}
impl FromJsValue for WordLevelTrainerOptions {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        if let Ok(options) = from.downcast::<JsObject>() {
            let mut builder = WordLevelTrainer::builder();

            if let Ok(size) = options.get(cx, "vocabSize") {
                if let Some(size) = Option::from_value(size, cx)? {
                    builder.vocab_size(size);
                }
            }
            if let Ok(freq) = options.get(cx, "minFrequency") {
                if let Some(freq) = Option::from_value(freq, cx)? {
                    builder.min_frequency(freq);
                }
            }
            if let Ok(tokens) = options.get(cx, "specialTokens") {
                if tokens.downcast::<JsNull>().is_err() && tokens.downcast::<JsUndefined>().is_err()
                {
                    builder.special_tokens(
                        tokens
                            .downcast::<JsArray>()
                            .map_err(|e| Error(format!("{}", e)))?
                            .to_vec(cx)?
                            .into_iter()
                            .map(|token| Ok(AddedToken::from_value(token, cx)?.into()))
                            .collect::<Result<Vec<_>, Error>>()?,
                    );
                }
            }
            if let Ok(show) = options.get(cx, "showProgress") {
                if let Some(show) = Option::from_value(show, cx)? {
                    builder.show_progress(show);
                }
            }

            Ok(Self(
                builder
                    .build()
                    .expect("WordLevelTrainerBuilder cannot fail"),
            ))
        } else {
            Err(Error("Expected options type: object".into()))
        }
    }
}

/// wordlevel_trainer(options?: {
///   vocabSize?: number = 30000,
///   minFrequency?: number = 0,
///   specialTokens?: string[] = [],
///   showProgress?: bool = true,
/// })
fn wordlevel_trainer(mut cx: FunctionContext) -> JsResult<JsTrainer> {
    let trainer = cx.extract_opt::<WordLevelTrainerOptions>(0)?.map_or_else(
        || WordLevelTrainer::builder().build().unwrap(),
        |o| o.into(),
    );

    let mut js_trainer = JsTrainer::new::<_, JsTrainer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_trainer.borrow_mut(&guard).trainer = Some(Arc::new(RwLock::new(trainer.into())));

    Ok(js_trainer)
}

// Unigram

struct UnigramTrainerOptions(UnigramTrainer);
impl From<UnigramTrainerOptions> for UnigramTrainer {
    fn from(v: UnigramTrainerOptions) -> Self {
        v.0
    }
}
impl FromJsValue for UnigramTrainerOptions {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        if let Ok(options) = from.downcast::<JsObject>() {
            let mut builder = UnigramTrainer::builder();

            if let Ok(size) = options.get(cx, "vocabSize") {
                if let Some(size) = Option::from_value(size, cx)? {
                    builder.vocab_size(size);
                }
            }
            if let Ok(nsub) = options.get(cx, "nSubIterations") {
                if let Some(nsub) = Option::from_value(nsub, cx)? {
                    builder.n_sub_iterations(nsub);
                }
            }
            if let Ok(factor) = options.get(cx, "shrinkingFactor") {
                if let Some(factor) = Option::from_value(factor, cx)? {
                    builder.shrinking_factor(factor);
                }
            }
            if let Ok(tokens) = options.get(cx, "specialTokens") {
                if tokens.downcast::<JsNull>().is_err() && tokens.downcast::<JsUndefined>().is_err()
                {
                    builder.special_tokens(
                        tokens
                            .downcast::<JsArray>()
                            .map_err(|e| Error(format!("{}", e)))?
                            .to_vec(cx)?
                            .into_iter()
                            .map(|token| Ok(AddedToken::from_value(token, cx)?.into()))
                            .collect::<Result<Vec<_>, Error>>()?,
                    );
                }
            }
            if let Ok(alphabet) = options.get(cx, "initialAlphabet") {
                if let Some(alphabet) = Option::from_value(alphabet, cx)? {
                    builder.initial_alphabet(alphabet);
                }
            }
            if let Ok(unk) = options.get(cx, "unkToken") {
                let unk = Option::from_value(unk, cx)?;
                builder.unk_token(unk);
            }
            if let Ok(max) = options.get(cx, "maxPieceLength") {
                if let Some(max) = Option::from_value(max, cx)? {
                    builder.max_piece_length(max);
                }
            }
            if let Ok(size) = options.get(cx, "seedSize") {
                if let Some(size) = Option::from_value(size, cx)? {
                    builder.seed_size(size);
                }
            }
            if let Ok(show) = options.get(cx, "showProgress") {
                if let Some(show) = Option::from_value(show, cx)? {
                    builder.show_progress(show);
                }
            }

            Ok(Self(builder.build()?))
        } else {
            Err(Error("Expected options type: object".into()))
        }
    }
}

/// unigram_trainer(options?: {
///  vocabSize?: number = 8000,
///  nSubIterations?: number = 2,
///  shrinkingFactor?: number = 0.75,
///  specialTokens?: string[] = [],
///  initialAlphabet?: string[] = [],
///  unkToken?: string = undefined,
///  maxPieceLength?: number = 16,
///  seedSize?: number = 1000000,
///  showProgress?: boolean = true,
/// })
fn unigram_trainer(mut cx: FunctionContext) -> JsResult<JsTrainer> {
    let trainer = cx
        .extract_opt::<UnigramTrainerOptions>(0)?
        .map_or_else(|| UnigramTrainer::builder().build().unwrap(), |o| o.into());

    let mut js_trainer = JsTrainer::new::<_, JsTrainer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_trainer.borrow_mut(&guard).trainer = Some(Arc::new(RwLock::new(trainer.into())));

    Ok(js_trainer)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BPETrainer", prefix), bpe_trainer)?;
    m.export_function(&format!("{}_WordPieceTrainer", prefix), wordpiece_trainer)?;
    m.export_function(&format!("{}_WordLevelTrainer", prefix), wordlevel_trainer)?;
    m.export_function(&format!("{}_UnigramTrainer", prefix), unigram_trainer)?;
    Ok(())
}
