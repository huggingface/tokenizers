#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![doc(html_favicon_url = "https://huggingface.co/favicon.ico")]
#![doc(html_logo_url = "https://huggingface.co/landing/assets/huggingface_logo.svg")]
#![doc = include_str!("../SPEC.md")]

pub(crate) mod json;
mod vendored;

/// Standard alphabet, padded on the way out and padding-*indifferent* on the way in, as `base64`
/// 0.13 was under `spm_precompiled`'s serde.
#[cfg(any(feature = "normalizers", all(test, feature = "deserialize")))]
pub(crate) const BASE64: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    base64::engine::GeneralPurposeConfig::new()
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent),
);

#[cfg(feature = "deserialize")]
mod from_json;

#[cfg(feature = "deserialize")]
pub use from_json::{from_json, from_json_file};

#[cfg(any(feature = "serialize", all(test, feature = "deserialize")))]
mod to_json;

#[cfg(feature = "serialize")]
pub use to_json::to_json;
