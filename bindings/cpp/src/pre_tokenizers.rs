use tk::normalizer::NormalizedString;
use tk::pre_tokenizers::{bert::BertPreTokenizer, PreTokenizerWrapper};
use tk::{PreTokenizedString, PreTokenizer, Token};

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
        include!("pre_tokenizers.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type Token;
        type NormalizedString;
        type PreTokenizedString;
        type PreTokenizerWrapper;
        type BertPreTokenizer;

        fn normalized_to_pre_tokenized_string(normalized: &NormalizedString) -> Box<PreTokenizedString>;
        fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString>;

        fn bert_pre_tokenizer() -> Box<BertPreTokenizer>;

        fn pre_tokenize_any(
            pre_tokenizer: &PreTokenizerWrapper,
            pre_tokenized: &mut PreTokenizedString,
        ) -> Result<()>;

        fn pre_tokenize_bert(
            pre_tokenizer: &BertPreTokenizer,
            pre_tokenized: &mut PreTokenizedString,
        ) -> Result<()>;

        fn get_splits(
            pre_tokenized: &PreTokenizedString,
            offset_ref: OffsetReferential,
            offset_type: OffsetType,
        ) -> Vec<Split>;
    }
}

fn normalized_to_pre_tokenized_string(normalized: &NormalizedString) -> Box<PreTokenizedString> {
    Box::new(normalized.clone().into())
}

fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString> {
    Box::new(str.into())
}

fn bert_pre_tokenizer() -> Box<BertPreTokenizer> {
    Box::new(BertPreTokenizer)
}

fn pre_tokenize_any(
    pre_tokenizer: &PreTokenizerWrapper,
    pre_tokenized: &mut PreTokenizedString,
) -> tk::Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}

fn pre_tokenize_bert(
    pre_tokenizer: &BertPreTokenizer,
    pre_tokenized: &mut PreTokenizedString,
) -> tk::Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}

fn get_splits(
    pre_tokenized: &PreTokenizedString,
    offset_ref: ffi::OffsetReferential,
    offset_type: ffi::OffsetType,
) -> Vec<ffi::Split> {
    let offset_ref = match offset_ref {
        ffi::OffsetReferential::Original => tk::OffsetReferential::Original,
        ffi::OffsetReferential::Normalized => tk::OffsetReferential::Normalized,
        _ => panic!("Illegal OffsetReferential value"),
    };
    let offset_type = match offset_type {
        ffi::OffsetType::Byte => tk::OffsetType::Byte,
        ffi::OffsetType::Char => tk::OffsetType::Char,
        _ => panic!("Illegal OffsetType value"),
    };
    pre_tokenized
        .get_splits(offset_ref, offset_type)
        .into_iter()
        .map(|(original, (start, end), _tokens)| ffi::Split {
            original: original.to_string(),
            start,
            end,
            // FIXME temporarily removed to work around a CXX conflict
            // has_tokens: tokens.is_some(),
            // tokens: tokens.as_ref().map_or_else(|| vec![], |v| v.clone()),
        }).collect()
}
