#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct OptionU32 {
        pub has_value: bool,
        pub value: u32,
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct OptionString {
        pub has_value: bool,
        pub value: String,
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct KVStringU32 {
        pub key: String,
        pub value: u32,
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct StringString {
        pub first: String,
        pub second: String,
    }

    extern "C++" {
        include!("tokenizers-cpp/models.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Token;
        type BpeBuilder;
        type Model;

        fn tokenize(model: &Model, sequence: &str) -> Result<Vec<Token>>;
        // `_model` suffix to avoid conflict with tokenizer.rs
        fn token_to_id_model(model: &Model, token: &str) -> OptionU32;
        fn id_to_token_model(model: &Model, id: u32) -> OptionString;
        fn get_vocab_model(model: &Model) -> Vec<KVStringU32>;
        fn get_vocab_size_model(model: &Model) -> usize;
        fn save(model: &Model, folder: &str, has_prefix: bool, prefix: &str)
            -> Result<Vec<String>>;

        fn bpe_builder() -> Box<BpeBuilder>;
        fn build_bpe(builder: &mut BpeBuilder) -> Result<Box<Model>>;
        fn files_bpe(builder: &mut BpeBuilder, vocab: String, merges: String);
        fn vocab_and_merges_bpe(
            builder: &mut BpeBuilder,
            vocab: Vec<KVStringU32>,
            merges: Vec<StringString>,
        );
        fn cache_capacity_bpe(builder: &mut BpeBuilder, capacity: usize);
        fn unk_token_bpe(builder: &mut BpeBuilder, unk_token: String);
        fn dropout_bpe(builder: &mut BpeBuilder, dropout: f32);
        fn continuing_subword_prefix_bpe(builder: &mut BpeBuilder, prefix: String);
        fn end_of_word_suffix_bpe(builder: &mut BpeBuilder, suffix: String);
        fn fuse_unk_bpe(builder: &mut BpeBuilder, fuse_unk: bool);
    }
}

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::wrap_option;
use derive_more::{Deref, DerefMut, From};
use ffi::*;
use tk::models::bpe::BpeBuilder as TkBpeBuilder;
use tk::{Model as ModelTrait, Result, Trainer as TrainerTrait};

#[derive(Deref, DerefMut, From)]
struct Token(tk::Token);

#[derive(Deref, DerefMut, From, Clone)]
pub struct Model(pub tk::ModelWrapper);

#[derive(Deref, DerefMut, From)]
pub struct Trainer(pub tk::models::TrainerWrapper);

impl ModelTrait for Model {
    type Trainer = crate::models::Trainer;

    fn tokenize(&self, sequence: &str) -> Result<Vec<tk::Token>> {
        self.0.tokenize(sequence)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.0.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.0.get_vocab_size()
    }

    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>> {
        self.0.save(folder, prefix)
    }

    fn get_trainer(&self) -> Self::Trainer {
        Trainer(self.0.get_trainer())
    }
}

impl TrainerTrait for Trainer {
    type Model = crate::models::Model;

    fn should_show_progress(&self) -> bool {
        self.0.should_show_progress()
    }

    fn train(&self, model: &mut Self::Model) -> Result<Vec<tk::AddedToken>> {
        self.0.train(model)
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.0.feed(iterator, process)
    }
}

fn tokenize(model: &Model, sequence: &str) -> Result<Vec<Token>> {
    model
        .tokenize(sequence)
        .map(|tokens| tokens.into_iter().map(|token| Token(token)).collect())
}

fn token_to_id_model(model: &Model, token: &str) -> OptionU32 {
    wrap_option!(model.token_to_id(token), OptionU32, 0)
}

fn id_to_token_model(model: &Model, id: u32) -> OptionString {
    wrap_option!(model.id_to_token(id), OptionString, "".to_string())
}

fn get_vocab_model(model: &Model) -> Vec<KVStringU32> {
    model
        .get_vocab()
        .iter()
        .map(|(k, v)| KVStringU32 {
            key: k.clone(),
            value: *v,
        })
        .collect()
}

fn get_vocab_size_model(model: &Model) -> usize {
    model.get_vocab_size()
}

fn save(model: &Model, folder: &str, has_prefix: bool, prefix: &str) -> Result<Vec<String>> {
    let prefix = if has_prefix { Some(prefix) } else { None };
    let mut error: Option<tk::Error> = None;
    let result = model.save(folder.as_ref(), prefix).map(|paths| {
        paths
            .into_iter()
            .filter_map(|p| match p.into_os_string().into_string() {
                Ok(x) => Some(x),
                Err(os_str) => {
                    error = Some(
                        format!("Path {} is not valid unicode", os_str.to_string_lossy()).into(),
                    );
                    None
                }
            })
            .collect()
    });
    if let Some(err) = error {
        Err(err)
    } else {
        result
    }
}

struct BpeBuilder(Option<TkBpeBuilder>);

fn bpe_builder() -> Box<BpeBuilder> {
    Box::new(BpeBuilder(Some(TkBpeBuilder::new())))
}

fn build_bpe(builder: &mut BpeBuilder) -> Result<Box<Model>> {
    match builder.0.take() {
        None => Err("Empty BpeBuilder".into()),
        Some(b) => Ok(Box::new(Model(b.build()?.into()))),
    }
}

fn files_bpe(builder: &mut BpeBuilder, vocab: String, merges: String) {
    builder.0 = builder.0.take().map(|b| b.files(vocab, merges));
}

fn vocab_and_merges_bpe(
    builder: &mut BpeBuilder,
    vocab: Vec<KVStringU32>,
    merges: Vec<StringString>,
) {
    let vocab = vocab.into_iter().map(|kv| (kv.key, kv.value)).collect();
    let merges = merges.into_iter().map(|ss| (ss.first, ss.second)).collect();
    builder.0 = builder.0.take().map(|b| b.vocab_and_merges(vocab, merges));
}

fn cache_capacity_bpe(builder: &mut BpeBuilder, capacity: usize) {
    builder.0 = builder.0.take().map(|b| b.cache_capacity(capacity));
}

fn unk_token_bpe(builder: &mut BpeBuilder, unk_token: String) {
    builder.0 = builder.0.take().map(|b| b.unk_token(unk_token));
}

fn dropout_bpe(builder: &mut BpeBuilder, dropout: f32) {
    builder.0 = builder.0.take().map(|b| b.dropout(dropout));
}
fn continuing_subword_prefix_bpe(builder: &mut BpeBuilder, prefix: String) {
    builder.0 = builder
        .0
        .take()
        .map(|b| b.continuing_subword_prefix(prefix));
}

fn end_of_word_suffix_bpe(builder: &mut BpeBuilder, suffix: String) {
    builder.0 = builder.0.take().map(|b| b.end_of_word_suffix(suffix));
}

fn fuse_unk_bpe(builder: &mut BpeBuilder, fuse_unk: bool) {
    builder.0 = builder.0.take().map(|b| b.fuse_unk(fuse_unk));
}
