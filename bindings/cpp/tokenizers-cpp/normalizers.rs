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

        fn normalized_string(str: &str) -> Box<NormalizedString>;

        fn bert_normalizer(
            clean_text: bool,
            handle_chinese_chars: bool,
            strip_accents: BertStripAccents,
            lowercase: bool,
        ) -> Box<Normalizer>;

        fn normalize(normalizer: &Normalizer, normalized: &mut NormalizedString) -> Result<()>;

        fn get_normalized(normalized: &NormalizedString) -> &str;
        fn get_original(normalized: &NormalizedString) -> &str;
    }
}

use derive_more::{Deref, DerefMut};
use tk::{normalizers::BertNormalizer, Normalizer as NormalizerTrait, Result};

#[derive(Deref, DerefMut)]
struct NormalizedString(tk::NormalizedString);

#[derive(Deref, DerefMut, Clone)]
pub struct Normalizer(pub tk::NormalizerWrapper);

impl NormalizerTrait for Normalizer {
    fn normalize(&self, normalized: &mut tk::NormalizedString) -> Result<()> {
        self.0.normalize(normalized)
    }
}

use ffi::BertStripAccents;

fn normalized_string(str: &str) -> Box<NormalizedString> {
    Box::new(NormalizedString(str.into()))
}

fn bert_normalizer(
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: BertStripAccents,
    lowercase: bool,
) -> Box<Normalizer> {
    let strip_accents = match strip_accents {
        BertStripAccents::False => Some(false),
        BertStripAccents::True => Some(true),
        BertStripAccents::DeterminedByLowercase => None,
        _ => None,
    };
    Box::new(Normalizer(
        BertNormalizer::new(clean_text, handle_chinese_chars, strip_accents, lowercase).into(),
    ))
}

fn normalize(normalizer: &Normalizer, normalized: &mut NormalizedString) -> Result<()> {
    normalizer.normalize(normalized)
}

fn get_normalized(normalized: &NormalizedString) -> &str {
    normalized.get()
}

fn get_original(normalized: &NormalizedString) -> &str {
    normalized.get_original()
}
