#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    #[namespace = "huggingface::tokenizers::ffi"]
    pub struct OptionUsize {
        pub has_value: bool,
        pub value: usize,
    }

    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone)]
    pub struct Offsets {
        pub start: usize,
        pub end: usize,
    }

    #[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone)]
    pub struct SpecialToken {
        pub token: String,
        pub id: u32,
    }

    extern "C++" {
        include!("tokenizers-cpp/processors.h");
        include!("tokenizers-cpp/models.h");

        #[namespace = "huggingface::tokenizers::ffi"]
        type OptionU32 = crate::models::ffi::OptionU32;
    }

    impl Vec<OptionU32> {}

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type PostProcessor;

        fn encoding_with_capacity(len: usize) -> Box<Encoding>;

        fn bert_post_processor(
            sep_token: &str,
            sep_id: u32,
            cls_token: &str,
            cls_id: u32,
        ) -> Box<PostProcessor>;

        fn byte_level_post_processor(
            add_prefix_space: bool,
            trim_offsets: bool,
        ) -> Box<PostProcessor>;

        fn roberta_post_processor(
            sep_token: &str,
            sep_id: u32,
            cls_token: &str,
            cls_id: u32,
            add_prefix_space: bool,
            trim_offsets: bool,
        ) -> Box<PostProcessor>;

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

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type TemplateProcessingBuilder;

        fn template_processing_builder() -> Box<TemplateProcessingBuilder>;

        fn single(&mut self, sequence_template: &str) -> Result<()>;
        fn pair(&mut self, sequence_template: &str) -> Result<()>;
        fn special_tokens(&mut self, tokens: &[SpecialToken]);
        fn build(&self) -> Result<Box<PostProcessor>>;
    }
}

use crate::wrap_option;
use derive_more::{Deref, DerefMut};
use ffi::*;
use tk::{
    processors::{
        bert::BertProcessing, byte_level::ByteLevel, roberta::RobertaProcessing,
        template::TemplateProcessingBuilder as TkTemplateProcessingBuilder,
    },
    PostProcessor as PostProcessorTrait, PostProcessorWrapper, Result,
};

// TODO may move Encoding to a separate module, but this depends on
//  the second part of https://github.com/dtolnay/cxx/issues/496
#[derive(Deref, DerefMut)]
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

#[derive(Deref, DerefMut, Clone)]
pub struct PostProcessor(pub PostProcessorWrapper);

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

fn make_post_processor<PP: Into<PostProcessorWrapper>>(post_processor: PP) -> Box<PostProcessor> {
    Box::new(PostProcessor(post_processor.into()))
}

fn bert_post_processor(
    sep_token: &str,
    sep_id: u32,
    cls_token: &str,
    cls_id: u32,
) -> Box<PostProcessor> {
    make_post_processor(BertProcessing::new(
        (sep_token.to_string(), sep_id),
        (cls_token.to_string(), cls_id),
    ))
}

fn byte_level_post_processor(add_prefix_space: bool, trim_offsets: bool) -> Box<PostProcessor> {
    make_post_processor(ByteLevel::new(add_prefix_space, trim_offsets))
}

fn roberta_post_processor(
    sep_token: &str,
    sep_id: u32,
    cls_token: &str,
    cls_id: u32,
    add_prefix_space: bool,
    trim_offsets: bool,
) -> Box<PostProcessor> {
    make_post_processor(
        RobertaProcessing::new(
            (sep_token.to_string(), sep_id),
            (cls_token.to_string(), cls_id),
        )
        .add_prefix_space(add_prefix_space)
        .trim_offsets(trim_offsets),
    )
}

struct TemplateProcessingBuilder(TkTemplateProcessingBuilder);
type TPResult<T> = std::result::Result<T, String>;

fn template_processing_builder() -> Box<TemplateProcessingBuilder> {
    Box::new(TemplateProcessingBuilder(
        TkTemplateProcessingBuilder::default(),
    ))
}

impl TemplateProcessingBuilder {
    fn single(&mut self, sequence_template: &str) -> TPResult<()> {
        self.0.try_single(sequence_template).map(|_| ())
    }

    fn pair(&mut self, sequence_template: &str) -> TPResult<()> {
        self.0.try_pair(sequence_template).map(|_| ())
    }

    fn special_tokens(&mut self, tokens: &[SpecialToken]) {
        self.0.special_tokens(
            tokens
                .iter()
                .map(|t| (t.token.as_str(), t.id))
                .collect::<Vec<(&str, u32)>>(),
        );
    }

    fn build(&self) -> TPResult<Box<PostProcessor>> {
        Ok(make_post_processor(self.0.build()?))
    }
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
