#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    // can't reuse from `models` because it contains String
    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct KVStringU32 {
        pub key: String,
        pub value: u32,
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct OptionU32 {
        pub has_value: bool,
        pub value: u32,
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct OptionUsize {
        pub has_value: bool,
        pub value: usize,
    }

    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Default, Debug, Clone)]
    pub struct Offsets {
        pub start: usize,
        pub end: usize,
    }

    extern "C++" {
        include!("tokenizers-cpp/processors.h");
        include!("tokenizers-cpp/tokens.h");
        include!("tokenizers-cpp/models.h");
        type Tokens = crate::tokens::ffi::Tokens;
    // FIXME still running into `Rust Vec containing C++ type is not supported yet`
    // #[namespace = "huggingface::tokenizers::ffi"]
    // type OptionU32 = crate::models::ffi::OptionU32;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
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

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Encoding;

        fn is_empty(&self) -> bool;
        fn len(&self) -> usize;
        fn n_sequences(&self) -> usize;
        fn get_tokens(&self) -> &[String];
        fn get_word_ids(&self) -> Vec<OptionU32>;
        fn get_sequence_ids(&self) -> Vec<OptionUsize>;
        fn get_ids(&self) -> &[u32];
        fn get_type_ids(&self) -> &[u32];
        fn get_offsets(&self) -> Vec<Offsets>;
        fn get_special_tokens_mask(&self) -> &[u32];
        fn get_attention_mask(&self) -> &[u32];
        fn get_overflowing(&self) -> Vec<Encoding>;
    }
}

use crate::wrap_option;
use derive_more::{Deref, DerefMut, From};
use ffi::*;
use tk::{
    processors::bert::BertProcessing as TkBertProcessing, PostProcessor as PostProcessorTrait,
    Result,
};

// TODO may move Encoding to a separate module, but this depends on
//  the second part of https://github.com/dtolnay/cxx/issues/496
#[derive(Deref, DerefMut, From)]
pub struct Encoding(pub tk::Encoding);

impl Encoding {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn n_sequences(&self) -> usize {
        self.0.n_sequences()
    }

    fn get_tokens(&self) -> &[String] {
        self.0.get_tokens()
    }

    fn get_word_ids(&self) -> Vec<OptionU32> {
        self.0
            .get_word_ids()
            .iter()
            .map(|x| wrap_option!(x, OptionU32, 0))
            .collect()
    }

    fn get_sequence_ids(&self) -> Vec<OptionUsize> {
        self.0
            .get_sequence_ids()
            .iter()
            .map(|x| wrap_option!(x, OptionUsize, 0))
            .collect()
    }

    fn get_ids(&self) -> &[u32] {
        self.0.get_ids()
    }

    fn get_type_ids(&self) -> &[u32] {
        self.0.get_type_ids()
    }

    fn get_offsets(&self) -> Vec<Offsets> {
        self.0
            .get_offsets()
            .iter()
            .map(|(start, end)| Offsets {
                start: *start,
                end: *end,
            })
            .collect()
    }

    fn get_special_tokens_mask(&self) -> &[u32] {
        self.0.get_special_tokens_mask()
    }

    fn get_attention_mask(&self) -> &[u32] {
        self.0.get_attention_mask()
    }

    fn get_overflowing(&self) -> Vec<Encoding> {
        self.0
            .get_overflowing()
            .iter()
            .map(|x| Encoding(x.clone()))
            .collect()
    }
}

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
