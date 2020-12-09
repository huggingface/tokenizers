#[cxx::bridge(namespace = "huggingface::tokenizers")]
pub mod ffi {
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
        include!("tokenizers-cpp/tokens.h");
        type Token = crate::tokens::ffi::Token;
        type Tokens = crate::tokens::ffi::Tokens;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Model;

        fn tokenize(model: &Model, sequence: &str) -> Result<Tokens>;
        // `_model` suffix to avoid conflict with tokenizer.rs
        fn token_to_id_model(model: &Model, token: &str) -> OptionU32;
        fn id_to_token_model(model: &Model, id: u32) -> OptionString;
        fn get_vocab_model(model: &Model) -> Vec<KVStringU32>;
        fn get_vocab_size_model(model: &Model) -> usize;
        fn save(model: &Model, folder: &str, has_prefix: bool, prefix: &str)
            -> Result<Vec<String>>;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type BpeBuilder;
        fn bpe_builder() -> Box<BpeBuilder>;
        fn build(&mut self) -> Result<Box<Model>>;
        fn files(&mut self, vocab: String, merges: String);
        fn vocab_and_merges(&mut self, vocab: Vec<KVStringU32>, merges: Vec<StringString>);
        fn cache_capacity(&mut self, capacity: usize);
        fn unk_token(&mut self, unk_token: String);
        fn dropout(&mut self, dropout: f32);
        fn continuing_subword_prefix(&mut self, prefix: String);
        fn end_of_word_suffix(&mut self, suffix: String);
        fn fuse_unk(&mut self, fuse_unk: bool);
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type WordPieceBuilder;
        fn word_piece_builder() -> Box<WordPieceBuilder>;
        fn build(&mut self) -> Result<Box<Model>>;
        fn vocab(&mut self, vocab: Vec<KVStringU32>);
        fn files(&mut self, vocab: &str);
        fn unk_token(&mut self, unk_token: &str);
        fn continuing_subword_prefix(&mut self, continuing_subword_prefix: &str);
        fn max_input_chars_per_word(&mut self, max_input_chars_per_word: usize);
    }
}

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::wrap_option;
use derive_more::{Deref, DerefMut, From};
use ffi::*;
use tk::{
    models::{bpe::BpeBuilder as TkBpeBuilder, wordpiece::WordPieceBuilder as TkWordPieceBuilder},
    Model as ModelTrait, Result, Trainer as TrainerTrait,
};

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

fn tokenize(model: &Model, sequence: &str) -> Result<Tokens> {
    Ok(model.tokenize(sequence)?.into())
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

fn update_builder<T, F: FnOnce(T) -> T>(opt: &mut Option<T>, f: F) {
    *opt = opt.take().map(f)
}

fn build<T, M: Into<tk::ModelWrapper>, F: FnOnce(T) -> Result<M>>(
    opt: &mut Option<T>,
    build_f: F,
) -> Result<Box<Model>> {
    match opt.take() {
        None => Err("Empty Builder".into()),
        Some(b) => Ok(Box::new(Model(build_f(b)?.into()))),
    }
}

type Vocab = HashMap<String, u32>;

fn make_vocab(entries: Vec<KVStringU32>) -> Vocab {
    entries.into_iter().map(|kv| (kv.key, kv.value)).collect()
}

#[derive(Deref, DerefMut)]
struct BpeBuilder(Option<TkBpeBuilder>);

fn bpe_builder() -> Box<BpeBuilder> {
    Box::new(BpeBuilder(Some(TkBpeBuilder::new())))
}

impl BpeBuilder {
    fn build(&mut self) -> Result<Box<Model>> {
        build(self, |b| b.build())
    }

    fn files(&mut self, vocab: String, merges: String) {
        update_builder(self, |b| b.files(vocab, merges));
    }

    fn vocab_and_merges(&mut self, vocab: Vec<KVStringU32>, merges: Vec<StringString>) {
        let merges = merges.into_iter().map(|ss| (ss.first, ss.second)).collect();
        update_builder(self, |b| b.vocab_and_merges(make_vocab(vocab), merges));
    }

    fn cache_capacity(&mut self, capacity: usize) {
        update_builder(self, |b| b.cache_capacity(capacity));
    }

    fn unk_token(&mut self, unk_token: String) {
        update_builder(self, |b| b.unk_token(unk_token));
    }

    fn dropout(&mut self, dropout: f32) {
        update_builder(self, |b| b.dropout(dropout));
    }
    fn continuing_subword_prefix(&mut self, prefix: String) {
        update_builder(self, |b| b.continuing_subword_prefix(prefix));
    }

    fn end_of_word_suffix(&mut self, suffix: String) {
        update_builder(self, |b| b.end_of_word_suffix(suffix));
    }

    fn fuse_unk(&mut self, fuse_unk: bool) {
        update_builder(self, |b| b.fuse_unk(fuse_unk));
    }
}

#[derive(Deref, DerefMut)]
struct WordPieceBuilder(Option<TkWordPieceBuilder>);

fn word_piece_builder() -> Box<WordPieceBuilder> {
    Box::new(WordPieceBuilder(Some(TkWordPieceBuilder::new())))
}

impl WordPieceBuilder {
    fn build(&mut self) -> Result<Box<Model>> {
        build(self, |b| b.build())
    }

    fn files(&mut self, vocab: &str) {
        update_builder(self, |b| b.files(vocab.to_string()))
    }

    fn vocab(&mut self, vocab: Vec<KVStringU32>) {
        update_builder(self, |b| b.vocab(make_vocab(vocab)))
    }

    fn unk_token(&mut self, unk_token: &str) {
        update_builder(self, |b| b.unk_token(unk_token.to_string()))
    }

    fn continuing_subword_prefix(&mut self, continuing_subword_prefix: &str) {
        update_builder(self, |b| {
            b.continuing_subword_prefix(continuing_subword_prefix.to_string())
        })
    }

    fn max_input_chars_per_word(&mut self, max_input_chars_per_word: usize) {
        update_builder(self, |b| {
            b.max_input_chars_per_word(max_input_chars_per_word)
        })
    }
}
