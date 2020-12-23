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

    pub struct TokenAndId {
        pub token: String,
        pub id: u32,
    }

    pub struct Merge {
        pub first: String,
        pub second: String,
    }

    pub struct UnigramEntry {
        pub token: String,
        pub log_prob: f64,
    }

    extern "C++" {
        include!("tokenizers-cpp/models.h");
        include!("tokenizers-cpp/tokens.h");
        type Token = crate::tokens::ffi::Token;
    }

    impl Vec<Token> {}

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Model;

        fn tokenize(model: &Model, sequence: &str) -> Result<Vec<Token>>;
        // `_model` suffix to avoid conflict with tokenizer.rs
        fn token_to_id_model(model: &Model, token: &str) -> OptionU32;
        fn id_to_token_model(model: &Model, id: u32) -> OptionString;
        fn get_vocab_model(model: &Model) -> Vec<TokenAndId>;
        fn get_vocab_size_model(model: &Model) -> usize;
        fn save(model: &Model, folder: &str, has_prefix: bool, prefix: &str)
            -> Result<Vec<String>>;

        fn unigram_model(
            vocab: &[UnigramEntry],
            has_unk_id: bool,
            unk_id: usize,
        ) -> Result<Box<Model>>;

        fn unigram_load_model(path: &str) -> Result<Box<Model>>;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type BpeBuilder;
        fn bpe_builder() -> Box<BpeBuilder>;
        fn build(&mut self) -> Result<Box<Model>>;
        fn files(&mut self, vocab: &str, merges: &str);
        fn vocab_and_merges(&mut self, vocab: &[TokenAndId], merges: &[Merge]);
        fn cache_capacity(&mut self, capacity: usize);
        fn unk_token(&mut self, unk_token: &str);
        fn dropout(&mut self, dropout: f32);
        fn continuing_subword_prefix(&mut self, prefix: &str);
        fn end_of_word_suffix(&mut self, suffix: &str);
        fn fuse_unk(&mut self, fuse_unk: bool);
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type WordPieceBuilder;
        fn word_piece_builder() -> Box<WordPieceBuilder>;
        fn build(&mut self) -> Result<Box<Model>>;
        fn vocab(&mut self, vocab: &[TokenAndId]);
        fn files(&mut self, vocab: &str);
        fn unk_token(&mut self, unk_token: &str);
        fn continuing_subword_prefix(&mut self, continuing_subword_prefix: &str);
        fn max_input_chars_per_word(&mut self, max_input_chars_per_word: usize);
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type WordLevelBuilder;
        fn word_level_builder() -> Box<WordLevelBuilder>;
        fn build(&mut self) -> Result<Box<Model>>;
        fn vocab(&mut self, vocab: &[TokenAndId]);
        fn files(&mut self, vocab: &str);
        fn unk_token(&mut self, unk_token: &str);
    }
}

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{tokens::wrap_tokens, wrap_option};
use derive_more::{Deref, DerefMut};
use ffi::*;
use tk::{
    models::{
        bpe::BpeBuilder as TkBpeBuilder, unigram::Unigram,
        wordlevel::WordLevelBuilder as TkWordLevelBuilder,
        wordpiece::WordPieceBuilder as TkWordPieceBuilder,
    },
    Model as ModelTrait, ModelWrapper, Result, Trainer as TrainerTrait,
};

#[derive(Deref, DerefMut, Clone)]
pub struct Model(pub ModelWrapper);

#[derive(Deref, DerefMut)]
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
    Ok(wrap_tokens(model.tokenize(sequence)?))
}

fn token_to_id_model(model: &Model, token: &str) -> OptionU32 {
    wrap_option!(model.token_to_id(token), OptionU32, 0)
}

fn id_to_token_model(model: &Model, id: u32) -> OptionString {
    wrap_option!(model.id_to_token(id), OptionString, "".to_string())
}

pub(crate) fn vocab_to_vec(vocab: Vocab) -> Vec<TokenAndId> {
    vocab
        .iter()
        .map(|(k, v)| TokenAndId {
            token: k.clone(),
            id: *v,
        })
        .collect()
}

fn get_vocab_model(model: &Model) -> Vec<TokenAndId> {
    vocab_to_vec(model.get_vocab())
}

fn get_vocab_size_model(model: &Model) -> usize {
    model.get_vocab_size()
}

fn some_if<T>(has_value: bool, value: T) -> Option<T> {
    if has_value {
        Some(value)
    } else {
        None
    }
}

fn save(model: &Model, folder: &str, has_prefix: bool, prefix: &str) -> Result<Vec<String>> {
    let prefix = some_if(has_prefix, prefix);
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

fn make_model<M: Into<ModelWrapper>>(model: Result<M>) -> Result<Box<Model>> {
    Ok(Box::new(Model(model?.into())))
}

fn unigram_model(vocab: &[UnigramEntry], has_unk_id: bool, unk_id: usize) -> Result<Box<Model>> {
    let vocab = vocab
        .iter()
        .map(|sf| (sf.token.clone(), sf.log_prob))
        .collect();
    let unk_id = some_if(has_unk_id, unk_id);
    make_model(Unigram::from(vocab, unk_id))
}

fn unigram_load_model(path: &str) -> Result<Box<Model>> {
    make_model(Unigram::load(path))
}

fn update_builder<T, F: FnOnce(T) -> T>(opt: &mut Option<T>, f: F) {
    *opt = opt.take().map(f)
}

fn build<T, M: Into<ModelWrapper>, F: FnOnce(T) -> Result<M>>(
    opt: &mut Option<T>,
    build_f: F,
) -> Result<Box<Model>> {
    match opt.take() {
        None => Err("Empty Builder".into()),
        Some(b) => make_model(build_f(b)),
    }
}

type Vocab = HashMap<String, u32>;

fn make_vocab(entries: &[TokenAndId]) -> Vocab {
    entries.iter().map(|kv| (kv.token.clone(), kv.id)).collect()
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

    fn files(&mut self, vocab: &str, merges: &str) {
        update_builder(self, |b| b.files(vocab.to_string(), merges.to_string()));
    }

    fn vocab_and_merges(&mut self, vocab: &[TokenAndId], merges: &[Merge]) {
        let merges = merges
            .iter()
            .map(|ss| (ss.first.clone(), ss.second.clone()))
            .collect();
        update_builder(self, |b| b.vocab_and_merges(make_vocab(vocab), merges));
    }

    fn cache_capacity(&mut self, capacity: usize) {
        update_builder(self, |b| b.cache_capacity(capacity));
    }

    fn unk_token(&mut self, unk_token: &str) {
        update_builder(self, |b| b.unk_token(unk_token.to_string()));
    }

    fn dropout(&mut self, dropout: f32) {
        update_builder(self, |b| b.dropout(dropout));
    }
    fn continuing_subword_prefix(&mut self, prefix: &str) {
        update_builder(self, |b| b.continuing_subword_prefix(prefix.to_string()));
    }

    fn end_of_word_suffix(&mut self, suffix: &str) {
        update_builder(self, |b| b.end_of_word_suffix(suffix.to_string()));
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

    fn vocab(&mut self, vocab: &[TokenAndId]) {
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

#[derive(Deref, DerefMut)]
struct WordLevelBuilder(Option<TkWordLevelBuilder>);

fn word_level_builder() -> Box<WordLevelBuilder> {
    Box::new(WordLevelBuilder(Some(TkWordLevelBuilder::new())))
}

impl WordLevelBuilder {
    fn build(&mut self) -> Result<Box<Model>> {
        build(self, |b| b.build())
    }

    fn files(&mut self, vocab: &str) {
        update_builder(self, |b| b.files(vocab.to_string()))
    }

    fn vocab(&mut self, vocab: &[TokenAndId]) {
        update_builder(self, |b| b.vocab(make_vocab(vocab)))
    }

    fn unk_token(&mut self, unk_token: &str) {
        update_builder(self, |b| b.unk_token(unk_token.to_string()))
    }
}
