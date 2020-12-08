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
        type PostProcessor;

        fn encoding_with_capacity(len: usize) -> Box<Encoding>;

        fn bert_post_processor(sep: KVStringU32, cls: KVStringU32) -> Box<PostProcessor>;

        fn added_tokens(post_processor: &PostProcessor, is_pair: bool) -> usize;
        fn process(
            post_processor: &PostProcessor,
            encoding: Box<Encoding>,
            add_special_tokens: bool,
        ) -> Result<Box<Encoding>>;
        fn process_pair(
            post_processor: &PostProcessor,
            encoding: Box<Encoding>,
            pair_encoding: Box<Encoding>,
            add_special_tokens: bool,
        ) -> Result<Box<Encoding>>;
    }
}

use derive_more::{Deref, DerefMut, From};
use ffi::*;
use tk::{
    processors::bert::BertProcessing as TkBertProcessing, PostProcessor as PostProcessorTrait,
    Result,
};

#[derive(Deref, DerefMut, From)]
pub struct Encoding(pub tk::Encoding);

#[derive(Deref, DerefMut, From, Clone)]
pub struct PostProcessor(pub tk::PostProcessorWrapper);

impl PostProcessorTrait for PostProcessor {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.0.added_tokens(is_pair)
    }

    fn process(
        &self,
        encoding: tk::Encoding,
        pair_encoding: Option<tk::Encoding>,
        add_special_tokens: bool,
    ) -> Result<tk::Encoding> {
        self.0.process(encoding, pair_encoding, add_special_tokens)
    }
}

fn encoding_with_capacity(len: usize) -> Box<Encoding> {
    Box::new(Encoding(tk::Encoding::with_capacity(len)))
}

fn bert_post_processor(sep: KVStringU32, cls: KVStringU32) -> Box<PostProcessor> {
    Box::new(PostProcessor(
        TkBertProcessing::new((sep.key, sep.value), (cls.key, cls.value)).into(),
    ))
}

fn added_tokens(post_processor: &PostProcessor, is_pair: bool) -> usize {
    post_processor.added_tokens(is_pair)
}

fn process(
    post_processor: &PostProcessor,
    encoding: Box<Encoding>,
    add_special_tokens: bool,
) -> Result<Box<Encoding>> {
    Ok(Box::new(Encoding(post_processor.process(
        (*encoding).0,
        None,
        add_special_tokens,
    )?)))
}

fn process_pair(
    post_processor: &PostProcessor,
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
