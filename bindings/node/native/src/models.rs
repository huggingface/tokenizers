extern crate tokenizers as tk;

use crate::container::Container;
use crate::extraction::*;
use crate::tasks::models::{BPEFromFilesTask, WordPieceFromFilesTask};
use neon::prelude::*;
use std::path::Path;
use tk::models::{bpe::BpeBuilder, wordpiece::WordPieceBuilder};

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
            // save(folder: string, name?: string)
            let folder = cx.extract::<String>(0)?;
            let name = cx.extract_opt::<String>(1)?;

            let this = cx.this();
            let guard = cx.lock();

            let files = this.borrow(&guard).model.execute(|model| {
                model.unwrap().save(Path::new(&folder), name.as_deref())
            }).map_err(|e| Error(format!("{}", e)))?;

            Ok(neon_serde::to_value(&mut cx, &files)?)
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct BpeOptions {
    cache_capacity: Option<usize>,
    dropout: Option<usize>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
}
impl BpeOptions {
    fn apply_to_bpe_builder(self, mut builder: BpeBuilder) -> BpeBuilder {
        if let Some(cache_capacity) = self.cache_capacity {
            builder = builder.cache_capacity(cache_capacity);
        }

        builder
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
    let vocab = cx.extract::<String>(0)?;
    let merges = cx.extract::<String>(1)?;

    let (options, callback) = match cx.extract_opt::<BpeOptions>(2) {
        // Options were there, and extracted
        Ok(Some(options)) => (options, cx.argument::<JsFunction>(3)?),
        // Options were undefined or null
        Ok(None) => (BpeOptions::default(), cx.argument::<JsFunction>(3)?),
        // Options not specified, callback instead
        Err(_) => (BpeOptions::default(), cx.argument::<JsFunction>(2)?),
    };

    let mut builder = tk::models::bpe::BPE::from_files(&vocab, &merges);
    builder = options.apply_to_bpe_builder(builder);

    let task = BPEFromFilesTask::new(builder);
    task.schedule(callback);
    Ok(cx.undefined())
}

/// bpe_empty()
pub fn bpe_empty(mut cx: FunctionContext) -> JsResult<JsModel> {
    let mut model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
    let bpe = tk::models::bpe::BPE::default();

    let guard = cx.lock();
    model.borrow_mut(&guard).model.make_owned(Box::new(bpe));

    Ok(model)
}

#[derive(Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct WordPieceOptions {
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    max_input_chars_per_word: Option<usize>,
}
impl WordPieceOptions {
    fn apply_to_wordpiece_builder(self, mut builder: WordPieceBuilder) -> WordPieceBuilder {
        if let Some(token) = self.unk_token {
            builder = builder.unk_token(token);
        }
        if let Some(prefix) = self.continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(prefix);
        }
        if let Some(max) = self.max_input_chars_per_word {
            builder = builder.max_input_chars_per_word(max);
        }

        builder
    }
}

/// wordpiece_from_files(vocab: String, options: {
///   unkToken?: String = "[UNK]",
///   maxInputCharsPerWord?: number = 100,
///   continuingSubwordPrefix?: "##",
/// }, callback)
pub fn wordpiece_from_files(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let vocab = cx.extract::<String>(0)?;

    let (options, callback) = match cx.extract_opt::<WordPieceOptions>(1) {
        // Options were there, and extracted
        Ok(Some(options)) => (options, cx.argument::<JsFunction>(2)?),
        // Options were undefined or null
        Ok(None) => (WordPieceOptions::default(), cx.argument::<JsFunction>(2)?),
        // Options not specified, callback instead
        Err(_) => (WordPieceOptions::default(), cx.argument::<JsFunction>(1)?),
    };

    let mut builder = tk::models::wordpiece::WordPiece::from_files(&vocab);
    builder = options.apply_to_wordpiece_builder(builder);

    let task = WordPieceFromFilesTask::new(builder);
    task.schedule(callback);
    Ok(cx.undefined())
}

/// wordpiece_empty()
pub fn wordpiece_empty(mut cx: FunctionContext) -> JsResult<JsModel> {
    let mut model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
    let wordpiece = tk::models::wordpiece::WordPiece::default();

    let guard = cx.lock();
    model
        .borrow_mut(&guard)
        .model
        .make_owned(Box::new(wordpiece));

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
