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
        // FIXME temporarily removed to work around a CXX conflict
        // has_tokens: bool,
        // NOTE CxxVector<Token> is not supported,
        //  &Vec<Token> is not supported in shared types
        // tokens: Vec<Token>,
    }

    extern "C++" {
        include!("tokenizers-cpp/pre_tokenizers.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Token;
        type NormalizedString;
        type PreTokenizedString;
        type PreTokenizerWrapper;

        // FIXME change to take Box<NormalizedString> after https://github.com/dtolnay/cxx/issues/496 is fixed
        fn normalized_to_pre_tokenized_string(
            normalized: &NormalizedString,
        ) -> Box<PreTokenizedString>;
        fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString>;

        fn pre_tokenize_any(
            pre_tokenizer: &PreTokenizerWrapper,
            pre_tokenized: &mut PreTokenizedString,
        ) -> Result<()>;

        fn pre_tokenize_bert(pre_tokenized: &mut PreTokenizedString) -> Result<()>;

        fn get_splits(
            pre_tokenized: &PreTokenizedString,
            offset_ref: OffsetReferential,
            offset_type: OffsetType,
        ) -> Vec<Split>;
    }
}

use derive_more::{Deref, DerefMut, From};
use tk::{pre_tokenizers::bert::BertPreTokenizer, PreTokenizer};

#[derive(Deref, DerefMut, From)]
struct NormalizedString(tk::NormalizedString);
#[derive(Deref, DerefMut, From)]
struct Token(tk::Token);
#[derive(Deref, DerefMut, From)]
struct PreTokenizedString(tk::PreTokenizedString);
#[derive(Deref, DerefMut, From)]
struct PreTokenizerWrapper(tk::pre_tokenizers::PreTokenizerWrapper);

fn normalized_to_pre_tokenized_string(normalized: &NormalizedString) -> Box<PreTokenizedString> {
    Box::new(PreTokenizedString(normalized.0.clone().into()))
}

fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString> {
    Box::new(PreTokenizedString(str.into()))
}

fn pre_tokenize_any(
    pre_tokenizer: &PreTokenizerWrapper,
    pre_tokenized: &mut PreTokenizedString,
) -> tk::Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}

fn pre_tokenize_bert(pre_tokenized: &mut PreTokenizedString) -> tk::Result<()> {
    BertPreTokenizer.pre_tokenize(pre_tokenized)
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
        .map(|(original, (start, end), _tokens)| ffi::Split {
            original: original.to_string(),
            start,
            end,
            // FIXME temporarily removed to work around a CXX conflict
            // has_tokens: tokens.is_some(),
            // tokens: tokens.as_ref().map_or_else(|| vec![], |v| v.clone()),
        })
        .collect()
}
