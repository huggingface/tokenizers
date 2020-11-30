use tk::models::bpe::{BpeBuilder as TkBpeBuilder, BPE};
use tk::{Model, Result, Token};

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
        type BPE;
        type BpeBuilder;

        fn tokenize_bpe(model: &BPE, sequence: &str) -> Result<Vec<Token>>;
        fn token_to_id_bpe(model: &BPE, token: &str) -> OptionU32;
        fn id_to_token_bpe(model: &BPE, id: u32) -> OptionString;
        fn get_vocab_bpe(model: &BPE) -> Vec<KVStringU32>;
        fn get_vocab_size_bpe(model: &BPE) -> usize;
        fn save_bpe(
            model: &BPE,
            folder: &str,
            has_prefix: bool,
            prefix: &str,
        ) -> Result<Vec<String>>;

        fn bpe_builder() -> Box<BpeBuilder>;
        fn build_bpe(builder: &mut BpeBuilder) -> Result<Box<BPE>>;
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

use ffi::*;

fn tokenize_bpe(model: &BPE, sequence: &str) -> Result<Vec<Token>> {
    model.tokenize(sequence)
}

fn token_to_id_bpe(model: &BPE, token: &str) -> OptionU32 {
    let id = model.token_to_id(token);
    OptionU32 {
        has_value: id.is_some(),
        value: id.unwrap_or(0),
    }
}

fn id_to_token_bpe(model: &BPE, id: u32) -> OptionString {
    let token = model.id_to_token(id);
    OptionString {
        has_value: token.is_some(),
        value: token.unwrap_or("".to_string()),
    }
}

fn get_vocab_bpe(model: &BPE) -> Vec<KVStringU32> {
    model
        .get_vocab()
        .iter()
        .map(|(k, v)| KVStringU32 {
            key: k.clone(),
            value: *v,
        })
        .collect()
}

fn get_vocab_size_bpe(model: &BPE) -> usize {
    model.get_vocab_size()
}

fn save_bpe(model: &BPE, folder: &str, has_prefix: bool, prefix: &str) -> Result<Vec<String>> {
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

fn build_bpe(builder: &mut BpeBuilder) -> Result<Box<BPE>> {
    match builder.0.take() {
        None => Err("Empty BpeBuilder".into()),
        Some(b) => Ok(Box::new(b.build()?)),
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
