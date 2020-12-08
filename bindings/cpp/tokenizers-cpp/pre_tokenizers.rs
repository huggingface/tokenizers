#[cxx::bridge(namespace = "huggingface::tokenizers")]
mod ffi {
    pub enum OffsetReferential {
        Original,
        Normalized,
    }

    pub enum OffsetType {
        Byte,
        Char,
    }

    pub struct Split {
        // NOTE may be changed when &str is supported in shared types
        original: String,
        start: usize,
        end: usize,
        has_tokens: bool,
        tokens: Tokens,
    }

    extern "C++" {
        include!("tokenizers-cpp/pre_tokenizers.h");
        include!("tokenizers-cpp/tokens.h");
        type Tokens = crate::models::ffi::Tokens;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type NormalizedString;
        type PreTokenizedString;
        type PreTokenizer;

        // FIXME change to take Box<NormalizedString> after https://github.com/dtolnay/cxx/issues/496 is fixed
        fn normalized_to_pre_tokenized_string(
            normalized: &NormalizedString,
        ) -> Box<PreTokenizedString>;
        fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString>;

        fn bert_pre_tokenizer() -> Box<PreTokenizer>;

        fn pre_tokenize(
            pre_tokenizer: &PreTokenizer,
            pre_tokenized: &mut PreTokenizedString,
        ) -> Result<()>;

        fn get_splits(
            pre_tokenized: &PreTokenizedString,
            offset_ref: OffsetReferential,
            offset_type: OffsetType,
        ) -> Vec<Split>;
    }
}

use derive_more::{Deref, DerefMut, From};
use ffi::*;
use tk::{pre_tokenizers::bert::BertPreTokenizer, PreTokenizer as PreTokenizerTrait, Result};

#[derive(Deref, DerefMut, From)]
struct NormalizedString(tk::NormalizedString);
#[derive(Deref, DerefMut, From)]
struct PreTokenizedString(tk::PreTokenizedString);
#[derive(Deref, DerefMut, From, Clone)]
pub struct PreTokenizer(pub tk::pre_tokenizers::PreTokenizerWrapper);

impl PreTokenizerTrait for PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut tk::PreTokenizedString) -> Result<()> {
        self.0.pre_tokenize(pretokenized)
    }
}

fn normalized_to_pre_tokenized_string(normalized: &NormalizedString) -> Box<PreTokenizedString> {
    Box::new(PreTokenizedString(normalized.0.clone().into()))
}

fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString> {
    Box::new(PreTokenizedString(str.into()))
}

fn bert_pre_tokenizer() -> Box<PreTokenizer> {
    Box::new(PreTokenizer(BertPreTokenizer.into()))
}

fn pre_tokenize(
    pre_tokenizer: &PreTokenizer,
    pre_tokenized: &mut PreTokenizedString,
) -> Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}

fn get_splits(
    pre_tokenized: &PreTokenizedString,
    offset_ref: ffi::OffsetReferential,
    offset_type: ffi::OffsetType,
) -> Vec<ffi::Split> {
    use crate::forward_cxx_enum;
    pre_tokenized
        .get_splits(
            forward_cxx_enum!(offset_ref, OffsetReferential, Original, Normalized),
            forward_cxx_enum!(offset_type, OffsetType, Byte, Char),
        )
        .into_iter()
        .map(|(original, (start, end), tokens)| ffi::Split {
            original: original.to_string(),
            start,
            end,
            has_tokens: tokens.is_some(),
            tokens: tokens
                .as_ref()
                .map_or_else(|| Tokens::default(), |v| v.into()),
        })
        .collect()
}
