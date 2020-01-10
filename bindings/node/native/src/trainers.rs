extern crate tokenizers as tk;

use crate::utils::Container;
use neon::prelude::*;
use std::collections::HashSet;

/// Trainer
pub struct Trainer {
    pub trainer: Container<dyn tk::tokenizer::Trainer>,
}

declare_types! {
    pub class JsTrainer for Trainer {
        init(_) {
            // This should not be called from JS
            Ok(Trainer {
                trainer: Container::Empty
            })
        }
    }
}

/// bpe_trainer(options?: {
///   vocabSize?: number = 30000,
///   minFrequency?: number = 2,
///   specialTokens?: string[] = [],
///   limitAlphabet?: number = undefined,
///   initialAlphabet?: string[] = [],
///   showProgress?: bool = true,
///   continuingSubwordPrefix?: string = undefined,
///   endOfWordSuffix?: string = undefined,
/// })
fn bpe_trainer(mut cx: FunctionContext) -> JsResult<JsTrainer> {
    let options = cx.argument_opt(0);

    let mut builder = tk::models::bpe::BpeTrainer::builder();

    if let Some(options) = options {
        if let Ok(options) = options.downcast::<JsObject>() {
            if let Ok(size) = options.get(&mut cx, "vocabSize") {
                builder = builder
                    .vocab_size(size.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize);
            }
            if let Ok(freq) = options.get(&mut cx, "minFrequency") {
                builder = builder
                    .min_frequency(freq.downcast::<JsNumber>().or_throw(&mut cx)?.value() as u32);
            }
            if let Ok(tokens) = options.get(&mut cx, "specialTokens") {
                builder = builder.special_tokens(
                    tokens
                        .downcast::<JsArray>()
                        .or_throw(&mut cx)?
                        .to_vec(&mut cx)?
                        .into_iter()
                        .map(|token| Ok(token.downcast::<JsString>().or_throw(&mut cx)?.value()))
                        .collect::<NeonResult<Vec<_>>>()?,
                );
            }
            if let Ok(limit) = options.get(&mut cx, "limitAlphabet") {
                builder = builder.limit_alphabet(
                    limit.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize,
                );
            }
            if let Ok(alphabet) = options.get(&mut cx, "initialAlphabet") {
                builder = builder.initial_alphabet(
                    alphabet
                        .downcast::<JsArray>()
                        .or_throw(&mut cx)?
                        .to_vec(&mut cx)?
                        .into_iter()
                        .map(|tokens| {
                            Ok(tokens
                                .downcast::<JsString>()
                                .or_throw(&mut cx)?
                                .value()
                                .chars()
                                .nth(0))
                        })
                        .collect::<NeonResult<Vec<_>>>()?
                        .into_iter()
                        .filter(|c| c.is_some())
                        .map(|c| c.unwrap())
                        .collect::<HashSet<_>>(),
                );
            }
            if let Ok(show) = options.get(&mut cx, "showProgress") {
                builder =
                    builder.show_progress(show.downcast::<JsBoolean>().or_throw(&mut cx)?.value());
            }
            if let Ok(prefix) = options.get(&mut cx, "continuingSubwordPrefix") {
                builder = builder.continuing_subword_prefix(
                    prefix.downcast::<JsString>().or_throw(&mut cx)?.value(),
                );
            }
            if let Ok(suffix) = options.get(&mut cx, "endOfWordSuffix") {
                builder = builder
                    .end_of_word_suffix(suffix.downcast::<JsString>().or_throw(&mut cx)?.value());
            }
        }
    }

    let mut trainer = JsTrainer::new::<_, JsTrainer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    trainer
        .borrow_mut(&guard)
        .trainer
        .to_owned(Box::new(builder.build()));
    Ok(trainer)
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
    let options = cx.argument_opt(0);

    let mut builder = tk::models::wordpiece::WordPieceTrainer::builder();

    if let Some(options) = options {
        if let Ok(options) = options.downcast::<JsObject>() {
            if let Ok(size) = options.get(&mut cx, "vocabSize") {
                builder = builder
                    .vocab_size(size.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize);
            }
            if let Ok(freq) = options.get(&mut cx, "minFrequency") {
                builder = builder
                    .min_frequency(freq.downcast::<JsNumber>().or_throw(&mut cx)?.value() as u32);
            }
            if let Ok(tokens) = options.get(&mut cx, "specialTokens") {
                builder = builder.special_tokens(
                    tokens
                        .downcast::<JsArray>()
                        .or_throw(&mut cx)?
                        .to_vec(&mut cx)?
                        .into_iter()
                        .map(|token| Ok(token.downcast::<JsString>().or_throw(&mut cx)?.value()))
                        .collect::<NeonResult<Vec<_>>>()?,
                );
            }
            if let Ok(limit) = options.get(&mut cx, "limitAlphabet") {
                builder = builder.limit_alphabet(
                    limit.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize,
                );
            }
            if let Ok(alphabet) = options.get(&mut cx, "initialAlphabet") {
                builder = builder.initial_alphabet(
                    alphabet
                        .downcast::<JsArray>()
                        .or_throw(&mut cx)?
                        .to_vec(&mut cx)?
                        .into_iter()
                        .map(|tokens| {
                            Ok(tokens
                                .downcast::<JsString>()
                                .or_throw(&mut cx)?
                                .value()
                                .chars()
                                .nth(0))
                        })
                        .collect::<NeonResult<Vec<_>>>()?
                        .into_iter()
                        .filter(|c| c.is_some())
                        .map(|c| c.unwrap())
                        .collect::<HashSet<_>>(),
                );
            }
            if let Ok(show) = options.get(&mut cx, "showProgress") {
                builder =
                    builder.show_progress(show.downcast::<JsBoolean>().or_throw(&mut cx)?.value());
            }
            if let Ok(prefix) = options.get(&mut cx, "continuingSubwordPrefix") {
                builder = builder.continuing_subword_prefix(
                    prefix.downcast::<JsString>().or_throw(&mut cx)?.value(),
                );
            }
            if let Ok(suffix) = options.get(&mut cx, "endOfWordSuffix") {
                builder = builder
                    .end_of_word_suffix(suffix.downcast::<JsString>().or_throw(&mut cx)?.value());
            }
        }
    }

    let mut trainer = JsTrainer::new::<_, JsTrainer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    trainer
        .borrow_mut(&guard)
        .trainer
        .to_owned(Box::new(builder.build()));
    Ok(trainer)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BPETrainer", prefix), bpe_trainer)?;
    m.export_function(&format!("{}_WordPieceTrainer", prefix), wordpiece_trainer)?;
    Ok(())
}
