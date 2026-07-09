use icu_normalizer::{ComposingNormalizerBorrowed, DecomposingNormalizerBorrowed};

pub(crate) static ICU_NFC: ComposingNormalizerBorrowed<'static> =
    ComposingNormalizerBorrowed::new_nfc();
pub(crate) static ICU_NFKC: ComposingNormalizerBorrowed<'static> =
    ComposingNormalizerBorrowed::new_nfkc();
pub(crate) static ICU_NFD: DecomposingNormalizerBorrowed<'static> =
    DecomposingNormalizerBorrowed::new_nfd();
pub(crate) static ICU_NFKD: DecomposingNormalizerBorrowed<'static> =
    DecomposingNormalizerBorrowed::new_nfkd();
