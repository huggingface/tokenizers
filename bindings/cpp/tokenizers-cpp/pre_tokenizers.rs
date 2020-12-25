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

    pub enum SplitDelimiterBehavior {
        Removed,
        Isolated,
        MergedWithPrevious,
        MergedWithNext,
        Contiguous,
    }

    pub struct Split {
        // NOTE may be changed when &str is supported in shared types
        original: String,
        start: usize,
        end: usize,
        has_tokens: bool,
        tokens: Vec<Token>,
    }

    extern "C++" {
        include!("tokenizers-cpp/pre_tokenizers.h");
        include!("tokenizers-cpp/tokens.h");
        type Token = crate::models::ffi::Token;
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type NormalizedString;
        type PreTokenizedString;
        type PreTokenizer;
        type PreTokenizerVec;

        // FIXME change to take Box<NormalizedString> after https://github.com/dtolnay/cxx/issues/496 is fixed
        fn normalized_to_pre_tokenized_string(
            normalized: &NormalizedString,
        ) -> Box<PreTokenizedString>;
        fn str_to_pre_tokenized_string(str: &str) -> Box<PreTokenizedString>;

        fn bert_pre_tokenizer() -> Box<PreTokenizer>;

        fn byte_level_pre_tokenizer(
            add_prefix_space: bool,
        ) -> Box<PreTokenizer>;

        // TODO should take char and return without Result, but see https://github.com/dtolnay/cxx/issues/592 (for metaspace as well)
        fn char_delimiter_pre_tokenizer(delimiter_cp: u32) -> Result<Box<PreTokenizer>>;

        fn metaspace_pre_tokenizer(
            replacement_cp: u32,
            add_prefix_space: bool,
        ) -> Result<Box<PreTokenizer>>;

        fn whitespace_pre_tokenizer() -> Box<PreTokenizer>;

        fn split_literal_pre_tokenizer(
            pattern: &str,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        ) -> Box<PreTokenizer>;

        fn split_regex_pre_tokenizer(
            pattern: &str,
            behavior: SplitDelimiterBehavior,
            invert: bool,
        ) -> Result<Box<PreTokenizer>>;

        fn punctuation_pre_tokenizer() -> Box<PreTokenizer>;

        fn whitespace_split_pre_tokenizer() -> Box<PreTokenizer>;

        fn digits_pre_tokenizer(individual_digits: bool) -> Box<PreTokenizer>;

        fn unicode_scripts_pre_tokenizer() -> Box<PreTokenizer>;

        fn init_pre_tokenizer_vec() -> Box<PreTokenizerVec>;

        fn add_pre_tokenizer(
            pre_tokenizers: &mut PreTokenizerVec,
            pre_tokenizer: Box<PreTokenizer>,
        );

        fn sequence_pre_tokenizer(pre_tokenizers: Box<PreTokenizerVec>) -> Box<PreTokenizer>;

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

use crate::{forward_cxx_enum, tokens::wrap_tokens_ref};
use derive_more::{Deref, DerefMut};
use ffi::*;
use tk::{
    pre_tokenizers::{
        bert::BertPreTokenizer,
        byte_level::ByteLevel,
        delimiter::CharDelimiterSplit,
        digits::Digits,
        metaspace::Metaspace,
        punctuation::Punctuation,
        sequence::Sequence,
        split::{Split as SplitPreTokenizer, SplitPattern},
        unicode_scripts::UnicodeScripts,
        whitespace::{Whitespace, WhitespaceSplit},
        PreTokenizerWrapper,
    },
    PreTokenizer as PreTokenizerTrait, Result,
};

#[derive(Deref, DerefMut)]
struct NormalizedString(tk::NormalizedString);

#[derive(Deref, DerefMut)]
struct PreTokenizedString(tk::PreTokenizedString);

#[derive(Deref, DerefMut, Clone)]
pub struct PreTokenizer(pub PreTokenizerWrapper);

#[derive(Deref, DerefMut, Clone)]
pub struct PreTokenizerVec(pub Vec<PreTokenizer>);

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

fn make_pre_tokenizer<PT: Into<PreTokenizerWrapper>>(pre_tokenizer: PT) -> Box<PreTokenizer> {
    Box::new(PreTokenizer(pre_tokenizer.into()))
}

fn bert_pre_tokenizer() -> Box<PreTokenizer> {
    make_pre_tokenizer(BertPreTokenizer)
}

fn byte_level_pre_tokenizer(add_prefix_space: bool) -> Box<PreTokenizer> {
    make_pre_tokenizer(ByteLevel::new(add_prefix_space, true))
}

pub fn u32_to_char(value: u32, name: &str) -> Result<char> {
    std::char::from_u32(value)
        .ok_or_else(|| format!("{} is invalid Unicode scalar value: {}", name, value).into())
}

fn char_delimiter_pre_tokenizer(delimiter_cp: u32) -> Result<Box<PreTokenizer>> {
    Ok(make_pre_tokenizer(CharDelimiterSplit::new(u32_to_char(
        delimiter_cp,
        "Delimiter",
    )?)))
}

fn metaspace_pre_tokenizer(
    replacement_cp: u32,
    add_prefix_space: bool,
) -> Result<Box<PreTokenizer>> {
    Ok(make_pre_tokenizer(Metaspace::new(
        u32_to_char(replacement_cp, "Replacement")?,
        add_prefix_space,
    )))
}

fn whitespace_pre_tokenizer() -> Box<PreTokenizer> {
    make_pre_tokenizer(Whitespace::default())
}

fn split_pre_tokenizer_helper(
    pattern: SplitPattern,
    behavior: SplitDelimiterBehavior,
    invert: bool,
) -> Result<SplitPreTokenizer> {
    SplitPreTokenizer::new(
        pattern,
        forward_cxx_enum!(
            behavior,
            SplitDelimiterBehavior,
            Removed,
            Isolated,
            MergedWithPrevious,
            MergedWithNext,
            Contiguous
        ),
        invert,
    )
}

fn split_literal_pre_tokenizer(
    pattern: &str,
    behavior: SplitDelimiterBehavior,
    invert: bool,
) -> Box<PreTokenizer> {
    make_pre_tokenizer(
        split_pre_tokenizer_helper(SplitPattern::String(pattern.to_string()), behavior, invert)
            .expect("Creating Split pre-tokenizer for a literal string should not fail"),
    )
}

fn split_regex_pre_tokenizer(
    pattern: &str,
    behavior: SplitDelimiterBehavior,
    invert: bool,
) -> Result<Box<PreTokenizer>> {
    Ok(make_pre_tokenizer(split_pre_tokenizer_helper(
        SplitPattern::Regex(pattern.to_string()),
        behavior,
        invert,
    )?))
}

fn punctuation_pre_tokenizer() -> Box<PreTokenizer> {
    make_pre_tokenizer(Punctuation)
}

fn whitespace_split_pre_tokenizer() -> Box<PreTokenizer> {
    make_pre_tokenizer(WhitespaceSplit)
}

fn digits_pre_tokenizer(individual_digits: bool) -> Box<PreTokenizer> {
    make_pre_tokenizer(Digits::new(individual_digits))
}

fn unicode_scripts_pre_tokenizer() -> Box<PreTokenizer> {
    make_pre_tokenizer(UnicodeScripts)
}

fn init_pre_tokenizer_vec() -> Box<PreTokenizerVec> {
    Box::new(PreTokenizerVec(vec![]))
}

fn add_pre_tokenizer(pre_tokenizers: &mut PreTokenizerVec, pre_tokenizer: Box<PreTokenizer>) {
    pre_tokenizers.push(*pre_tokenizer)
}

fn sequence_pre_tokenizer(pre_tokenizers: Box<PreTokenizerVec>) -> Box<PreTokenizer> {
    make_pre_tokenizer(Sequence::new(
        (*pre_tokenizers).0.into_iter().map(|n| n.0).collect(),
    ))
}

fn pre_tokenize(
    pre_tokenizer: &PreTokenizer,
    pre_tokenized: &mut PreTokenizedString,
) -> Result<()> {
    pre_tokenizer.pre_tokenize(pre_tokenized)
}

fn get_splits(
    pre_tokenized: &PreTokenizedString,
    offset_ref: OffsetReferential,
    offset_type: OffsetType,
) -> Vec<Split> {
    pre_tokenized
        .get_splits(
            forward_cxx_enum!(offset_ref, OffsetReferential, Original, Normalized),
            forward_cxx_enum!(offset_type, OffsetType, Byte, Char),
        )
        .into_iter()
        .map(|(original, (start, end), tokens)| Split {
            original: original.to_string(),
            start,
            end,
            has_tokens: tokens.is_some(),
            tokens: tokens.as_ref().map_or_else(|| vec![], wrap_tokens_ref),
        })
        .collect()
}
