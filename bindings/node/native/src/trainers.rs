extern crate tokenizers as tk;

use crate::extraction::*;
use crate::models::Model;
use crate::tokenizer::AddedToken;
use neon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use tk::models::{bpe::BpeTrainer, wordpiece::WordPieceTrainer, TrainerWrapper};

/// Trainer
#[derive(Clone)]
pub struct Trainer {
    pub trainer: Option<Arc<TrainerWrapper>>,
}

impl tk::Trainer for Trainer {
    type Model = Model;

    fn should_show_progress(&self) -> bool {
        self.trainer
            .as_ref()
            .expect("Uninitialized Trainer")
            .should_show_progress()
    }

    fn train(&self, words: HashMap<String, u32>) -> tk::Result<(Self::Model, Vec<tk::AddedToken>)> {
        let (model, special_tokens) = self
            .trainer
            .as_ref()
            .ok_or("Uninitialized Trainer")?
            .train(words)?;

        Ok((model.into(), special_tokens))
    }

    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        self.trainer
            .as_ref()
            .expect("Uninitialized Trainer")
            .process_tokens(words, tokens)
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
    js_trainer.borrow_mut(&guard).trainer = Some(Arc::new(trainer.into()));

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
    js_trainer.borrow_mut(&guard).trainer = Some(Arc::new(trainer.into()));

    Ok(js_trainer)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BPETrainer", prefix), bpe_trainer)?;
    m.export_function(&format!("{}_WordPieceTrainer", prefix), wordpiece_trainer)?;
    Ok(())
}
