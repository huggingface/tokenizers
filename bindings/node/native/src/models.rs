extern crate tokenizers as tk;

use crate::container::Container;
use crate::tasks::models::{BPEFromFilesTask, WordPieceFromFilesTask};
use neon::prelude::*;
use std::path::Path;

/// Model
pub struct Model {
    pub model: Container<dyn tk::tokenizer::Model>,
}

declare_types! {
    pub class JsModel for Model {
        init(_) {
            // This should not be called from JS
            Ok(Model {
                model: Container::Empty
            })
        }

        method save(mut cx) {
            /// save(folder: string, name?: string)
            let folder = cx.argument::<JsString>(0)?.value();

            let name = if let Some(name_arg) = cx.argument_opt(1) {
                if name_arg.downcast::<JsUndefined>().is_err() {
                    Some(name_arg.downcast_or_throw::<JsString, _>(&mut cx)?.value())
                } else {
                    None
                }
            } else {
                None
            };

            let this = cx.this();
            let guard = cx.lock();
            let result = this.borrow(&guard).model.execute(|model| {
                model.unwrap().save(Path::new(&folder), name.as_deref())
            });

            match result {
                Ok(r) => {
                    let array = JsArray::new(&mut cx, r.len() as u32);
                    for (i, path) in r.into_iter().enumerate() {
                        let n = JsString::new(&mut cx, path.to_string_lossy().into_owned());
                        array.set(&mut cx, i as u32, n)?;
                    }
                    Ok(array.upcast())
                },
                Err(e) => cx.throw_error(format!("{}", e))
            }
        }
    }
}

/// bpe_from_files(vocab: String, merges: String, options: {
///   cacheCapacity?: number,
///   dropout?: number,
///   unkToken?: String,
///   continuingSubwordPrefix?: String,
///   endOfWordSuffix?: String
/// }, callback)
pub fn bpe_from_files(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let vocab = cx.argument::<JsString>(0)?.value() as String;
    let merges = cx.argument::<JsString>(1)?.value() as String;
    let options: Option<Handle<JsObject>>;
    let callback: Handle<JsFunction>;

    if cx.len() == 3 {
        options = None;
        callback = cx.argument::<JsFunction>(2)?;
    } else {
        options = Some(cx.argument::<JsObject>(2)?);
        callback = cx.argument::<JsFunction>(3)?;
    };

    let mut builder = tk::models::bpe::BPE::from_files(&vocab, &merges);

    if let Some(options) = options {
        if let Ok(options) = options.downcast::<JsObject>() {
            if let Ok(cache_capacity) = options.get(&mut cx, "cacheCapacity") {
                if cache_capacity.downcast::<JsUndefined>().is_err() {
                    let cache_capacity = cache_capacity
                        .downcast::<JsNumber>()
                        .or_throw(&mut cx)?
                        .value() as usize;
                    builder = builder.cache_capacity(cache_capacity);
                }
            }
            if let Ok(dropout) = options.get(&mut cx, "dropout") {
                if dropout.downcast::<JsUndefined>().is_err() {
                    let dropout = dropout.downcast::<JsNumber>().or_throw(&mut cx)?.value() as f32;
                    builder = builder.dropout(dropout);
                }
            }
            if let Ok(unk_token) = options.get(&mut cx, "unkToken") {
                if unk_token.downcast::<JsUndefined>().is_err() {
                    let unk_token =
                        unk_token.downcast::<JsString>().or_throw(&mut cx)?.value() as String;
                    builder = builder.unk_token(unk_token);
                }
            }
            if let Ok(prefix) = options.get(&mut cx, "continuingSubwordPrefix") {
                if prefix.downcast::<JsUndefined>().is_err() {
                    let prefix = prefix.downcast::<JsString>().or_throw(&mut cx)?.value() as String;
                    builder = builder.continuing_subword_prefix(prefix);
                }
            }
            if let Ok(suffix) = options.get(&mut cx, "endOfWordSuffix") {
                if suffix.downcast::<JsUndefined>().is_err() {
                    let suffix = suffix.downcast::<JsString>().or_throw(&mut cx)?.value() as String;
                    builder = builder.end_of_word_suffix(suffix);
                }
            }
        }
    }

    let task = BPEFromFilesTask::new(builder);
    task.schedule(callback);
    Ok(cx.undefined())
}

/// bpe_empty()
pub fn bpe_empty(mut cx: FunctionContext) -> JsResult<JsModel> {
    let mut model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
    let bpe = tk::models::bpe::BPE::default();

    let guard = cx.lock();
    model.borrow_mut(&guard).model.to_owned(Box::new(bpe));

    Ok(model)
}

/// wordpiece_from_files(vocab: String, options: {
///   unkToken?: String = "[UNK]",
///   maxInputCharsPerWord?: number = 100,
///   continuingSubwordPrefix?: "##",
/// }, callback)
pub fn wordpiece_from_files(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let vocab = cx.argument::<JsString>(0)?.value() as String;
    let options: Option<Handle<JsObject>>;
    let callback: Handle<JsFunction>;

    if cx.len() == 2 {
        options = None;
        callback = cx.argument::<JsFunction>(1)?;
    } else {
        options = Some(cx.argument::<JsObject>(1)?);
        callback = cx.argument::<JsFunction>(2)?;
    };

    let mut builder = tk::models::wordpiece::WordPiece::from_files(&vocab);

    if let Some(options) = options {
        if let Ok(options) = options.downcast::<JsObject>() {
            if let Ok(unk) = options.get(&mut cx, "unkToken") {
                if unk.downcast::<JsUndefined>().is_err() {
                    builder = builder
                        .unk_token(unk.downcast::<JsString>().or_throw(&mut cx)?.value() as String);
                }
            }
            if let Ok(max) = options.get(&mut cx, "maxInputCharsPerWord") {
                if max.downcast::<JsUndefined>().is_err() {
                    builder = builder.max_input_chars_per_word(
                        max.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize,
                    );
                }
            }
            if let Ok(prefix) = options.get(&mut cx, "continuingSubwordPrefix") {
                if prefix.downcast::<JsUndefined>().is_err() {
                    builder = builder.continuing_subword_prefix(
                        prefix.downcast::<JsString>().or_throw(&mut cx)?.value() as String,
                    );
                }
            }
        }
    }

    let task = WordPieceFromFilesTask::new(builder);
    task.schedule(callback);
    Ok(cx.undefined())
}

/// wordpiece_empty()
pub fn wordpiece_empty(mut cx: FunctionContext) -> JsResult<JsModel> {
    let mut model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
    let wordpiece = tk::models::wordpiece::WordPiece::default();

    let guard = cx.lock();
    model.borrow_mut(&guard).model.to_owned(Box::new(wordpiece));

    Ok(model)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BPE_from_files", prefix), bpe_from_files)?;
    m.export_function(&format!("{}_BPE_empty", prefix), bpe_empty)?;
    m.export_function(
        &format!("{}_WordPiece_from_files", prefix),
        wordpiece_from_files,
    )?;
    m.export_function(&format!("{}_WordPiece_empty", prefix), wordpiece_empty)?;
    Ok(())
}
