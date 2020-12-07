#[cxx::bridge(namespace = "huggingface::tokenizers")]
pub mod ffi {
    pub enum BertStripAccents {
        DeterminedByLowercase,
        False,
        True,
    }

    extern "C++" {
        include!("tokenizers-cpp/normalizers.h");
    }

    #[namespace = "huggingface::tokenizers::ffi"]
    extern "Rust" {
        type NormalizedString;
        type Normalizer;
        type BertNormalizer;

        fn normalized_string(str: &str) -> Box<NormalizedString>;

        fn bert_normalizer_wrapper(
            clean_text: bool,
            handle_chinese_chars: bool,
            strip_accents: BertStripAccents,
            lowercase: bool,
        ) -> Box<Normalizer>;

        fn bert_normalizer(
            clean_text: bool,
            handle_chinese_chars: bool,
            strip_accents: BertStripAccents,
            lowercase: bool,
        ) -> Box<BertNormalizer>;

        fn normalize_any(normalizer: &Normalizer, normalized: &mut NormalizedString) -> Result<()>;

        fn normalize_bert(
            normalizer: &BertNormalizer,
            normalized: &mut NormalizedString,
        ) -> Result<()>;

        fn get_normalized(normalized: &NormalizedString) -> &str;
        fn get_original(normalized: &NormalizedString) -> &str;
    }
}

use derive_more::{Deref, DerefMut, From};
use tk::{normalizers::BertNormalizer as TkBertNormalizer, Normalizer as NormalizerTrait, Result};

#[derive(Deref, DerefMut, From)]
struct NormalizedString(tk::NormalizedString);

#[derive(Deref, DerefMut, From, Clone)]
pub struct Normalizer(pub tk::NormalizerWrapper);

impl NormalizerTrait for Normalizer {
    fn normalize(&self, normalized: &mut tk::NormalizedString) -> Result<()> {
        self.0.normalize(normalized)
    }
}

#[derive(Deref, DerefMut, From)]
struct BertNormalizer(TkBertNormalizer);

use ffi::BertStripAccents;

fn normalized_string(str: &str) -> Box<NormalizedString> {
    Box::new(NormalizedString(str.into()))
}

fn tk_bert_normalizer(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: BertStripAccents,
    lowercase: bool,
) -> TkBertNormalizer {
    let strip_accents = match strip_accents {
        BertStripAccents::False => Some(false),
        BertStripAccents::True => Some(true),
        BertStripAccents::DeterminedByLowercase => None,
        _ => None,
    };
    TkBertNormalizer::new(clean_text, handle_chinese_chars, strip_accents, lowercase)
}

fn bert_normalizer_wrapper(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: BertStripAccents,
    lowercase: bool,
) -> Box<Normalizer> {
    Box::new(Normalizer(
        tk_bert_normalizer(clean_text, handle_chinese_chars, strip_accents, lowercase).into(),
    ))
}

fn bert_normalizer(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: BertStripAccents,
    lowercase: bool,
) -> Box<BertNormalizer> {
    Box::new(BertNormalizer(tk_bert_normalizer(
        clean_text,
        handle_chinese_chars,
        strip_accents,
        lowercase,
    )))
}

fn normalize_any(normalizer: &Normalizer, normalized: &mut NormalizedString) -> Result<()> {
    normalizer.normalize(normalized)
}

fn normalize_bert(normalizer: &BertNormalizer, normalized: &mut NormalizedString) -> Result<()> {
    normalizer.normalize(normalized)
}

fn get_normalized(normalized: &NormalizedString) -> &str {
    normalized.get()
}

fn get_original(normalized: &NormalizedString) -> &str {
    normalized.get_original()
}
