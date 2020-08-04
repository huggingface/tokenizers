extern crate tokenizers as tk;

use crate::extraction::*;
use crate::tasks::models::{BPEFromFilesTask, WordPieceFromFilesTask};
use neon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use tk::models::{bpe::BpeBuilder, wordpiece::WordPieceBuilder, ModelWrapper};
use tk::Model as ModelTrait;
use tk::Token;

/// Model
#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    #[serde(flatten)]
    pub model: Option<Arc<ModelWrapper>>,
}

impl From<ModelWrapper> for Model {
    fn from(wrapper: ModelWrapper) -> Self {
        Self {
            model: Some(Arc::new(wrapper)),
        }
    }
}

impl tk::Model for Model {
    fn tokenize(&self, sequence: &str) -> tk::Result<Vec<Token>> {
        self.model
            .as_ref()
            .ok_or("Uninitialized Model")?
            .tokenize(sequence)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.as_ref()?.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.model.as_ref()?.id_to_token(id)
    }

    fn get_vocab(&self) -> &HashMap<String, u32> {
        self.model
            .as_ref()
            .expect("Uninitialized Model")
            .get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.model
            .as_ref()
            .expect("Uninitialized Model")
            .get_vocab_size()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> tk::Result<Vec<PathBuf>> {
        self.model
            .as_ref()
            .ok_or("Uninitialized Model")?
            .save(folder, name)
    }
}

declare_types! {
    pub class JsModel for Model {
        init(_) {
            // This should not be called from JS
            Ok(Model { model: None })
        }

        method save(mut cx) {
            // save(folder: string, name?: string)
            let folder = cx.extract::<String>(0)?;
            let name = cx.extract_opt::<String>(1)?;

            let this = cx.this();
            let guard = cx.lock();

            let files = this.borrow(&guard)
                .model.as_ref().unwrap()
                .save(
                    Path::new(&folder),
                    name.as_deref()
                )
                .map_err(|e| Error(format!("{}", e)))?;

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
    model.borrow_mut(&guard).model = Some(Arc::new(bpe.into()));

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
    model.borrow_mut(&guard).model = Some(Arc::new(wordpiece.into()));

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
