#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct KVStringU32 {
        pub key: String,
        pub value: u32,
    }

    extern "C++" {
        include!("tokenizers-cpp/processors.h");
    // include!("tokenizers-cpp/models.rs.h");

    // type KVStringU32 = crate::models::ffi::KVStringU32;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Encoding;
        type PostProcessorWrapper;
        type BertProcessing;

        fn encoding_with_capacity(len: usize) -> Box<Encoding>;

        fn bert_post_processor(sep: KVStringU32, cls: KVStringU32) -> Box<BertProcessing>;

        fn added_tokens_bert(post_processor: &BertProcessing, is_pair: bool) -> usize;
        fn process_bert(
            post_processor: &BertProcessing,
            encoding: Box<Encoding>,
            add_special_tokens: bool,
        ) -> Result<Box<Encoding>>;
        fn process_pair_bert(
            post_processor: &BertProcessing,
            encoding: Box<Encoding>,
            pair_encoding: Box<Encoding>,
            add_special_tokens: bool,
        ) -> Result<Box<Encoding>>;
    }
}

use derive_more::{Deref, DerefMut, From};
use ffi::*;
use tk::{processors::bert::BertProcessing as TkBertProcessing, PostProcessor, Result};

#[derive(Deref, DerefMut, From)]
struct Encoding(tk::Encoding);
#[derive(Deref, DerefMut, From)]
struct PostProcessorWrapper(tk::PostProcessorWrapper);
#[derive(Deref, DerefMut, From)]
struct BertProcessing(TkBertProcessing);

fn encoding_with_capacity(len: usize) -> Box<Encoding> {
    Box::new(Encoding(tk::Encoding::with_capacity(len)))
}

fn bert_post_processor(sep: KVStringU32, cls: KVStringU32) -> Box<BertProcessing> {
    Box::new(BertProcessing(TkBertProcessing::new(
        (sep.key, sep.value),
        (cls.key, cls.value),
    )))
}

fn added_tokens_bert(post_processor: &BertProcessing, is_pair: bool) -> usize {
    post_processor.added_tokens(is_pair)
}

fn process_bert(
    post_processor: &BertProcessing,
    encoding: Box<Encoding>,
    add_special_tokens: bool,
) -> Result<Box<Encoding>> {
    Ok(Box::new(Encoding(post_processor.process(
        (*encoding).0,
        None,
        add_special_tokens,
    )?)))
}

fn process_pair_bert(
    post_processor: &BertProcessing,
    encoding: Box<Encoding>,
    pair_encoding: Box<Encoding>,
    add_special_tokens: bool,
) -> Result<Box<Encoding>> {
    Ok(Box::new(Encoding(post_processor.process(
        (*encoding).0,
        Some((*pair_encoding).0),
        add_special_tokens,
    )?)))
}
