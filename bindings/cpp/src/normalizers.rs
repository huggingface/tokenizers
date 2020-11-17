use tk::normalizer::NormalizedString;
use tk::normalizers::{BertNormalizer, NormalizerWrapper};
use tk::Normalizer;

#[cxx::bridge(namespace = "huggingface::tokenizers::ffi")]
mod ffi {
    #[namespace = "huggingface::tokenizers"]
    pub enum BertStripAccents {
        DeterminedByLowercase,
        False,
        True,
    }

    extern "C++" {
        include!("normalizers.h");
    }

    extern "Rust" {
        type NormalizedString;
        type NormalizerWrapper;
        type BertNormalizer;

        fn normalized_string(str: &CxxString) -> Result<Box<NormalizedString>>;

        fn bert_normalizer(
            clean_text: bool,
            handle_chinese_chars: bool,
            strip_accents: BertStripAccents,
            lowercase: bool,
        ) -> Box<BertNormalizer>;

        fn normalize_any(
            normalizer: &NormalizerWrapper,
            normalized: &mut NormalizedString,
        ) -> Result<()>;

        fn normalize_bert(
            normalizer: &BertNormalizer,
            normalized: &mut NormalizedString,
        ) -> Result<()>;
    }
}

use cxx::CxxString;
use ffi::BertStripAccents;
use std::str::Utf8Error;

fn normalized_string(str: &CxxString) -> Result<Box<NormalizedString>, Utf8Error> {
    Ok(Box::new(NormalizedString::from(str.to_str()?)))
}

fn bert_normalizer(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: BertStripAccents,
    lowercase: bool,
) -> Box<BertNormalizer> {
    let strip_accents = match strip_accents {
        BertStripAccents::DeterminedByLowercase => None,
        BertStripAccents::False => Some(false),
        BertStripAccents::True => Some(true),
        _ => None,
    };
    Box::new(BertNormalizer::new(
        clean_text,
        handle_chinese_chars,
        strip_accents,
        lowercase,
    ))
}

fn normalize_any(
    normalizer: &NormalizerWrapper,
    normalized: &mut NormalizedString,
) -> tk::Result<()> {
    normalizer.normalize(normalized)
}

fn normalize_bert(
    normalizer: &BertNormalizer,
    normalized: &mut NormalizedString,
) -> tk::Result<()> {
    normalizer.normalize(normalized)
}
