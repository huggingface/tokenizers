pub(crate) mod cache;
// `pub` because `Tokenizer::from_pretrained` — the only caller — lives in `tk-convert`.
#[cfg(feature = "http")]
pub mod from_pretrained;
pub(crate) mod word_cache;

// Optional system-regex backend, needed only for a *regex* pattern that atomsplit does not cover.
// With `fancy-regex` off a stub compiles and those patterns error at load. Everything else works
// regardless: the atomsplit-native pre-tokenizers, and any `Split` or `Replace` whose pattern is a
// plain string (searched for directly, see `atomsplit::literal`).
#[cfg(feature = "fancy-regex")]
mod fancy;
#[cfg(feature = "fancy-regex")]
pub use fancy::SysRegex;
#[cfg(not(feature = "fancy-regex"))]
mod no_regex;
#[cfg(not(feature = "fancy-regex"))]
pub use no_regex::SysRegex;

// Recognize known GPT pre-tokenization regexes and route them to atomsplit's native (unrolled) FSM.
mod unrolled_regex;
pub use unrolled_regex::{DEEPSEEK_PATTERNS, GptFsm, GptFsmPattern, gpt_fsm, is_deepseek};

pub mod byte_level;
pub mod iter;
pub mod padding;
#[cfg(feature = "parallelism")]
pub mod parallelism;
pub mod progress;
pub mod search;
pub mod truncation;

// Re-export ProgressFormat for public API
pub use progress::ProgressFormat;

/// Serialize an [`AHashMap`](ahash::AHashMap) through a `BTreeMap` so the output has a
/// deterministic key order.
///
/// Only ever named from a `#[serde(serialize_with = ...)]` — today just the `TemplateProcessing`
/// special-token table. A hash map's iteration order is unspecified, and a `tokenizer.json` that
/// reorders its keys between two saves of the same tokenizer is a diff nobody wants to read.
#[cfg(feature = "serde")]
pub(crate) fn ordered_map<S, K, V>(
    value: &ahash::AHashMap<K, V>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
    K: serde::Serialize + std::cmp::Ord,
    V: serde::Serialize,
{
    use serde::Serialize;
    let ordered: std::collections::BTreeMap<_, _> = value.iter().collect();
    ordered.serialize(serializer)
}

/// Declare a pipeline component struct, or a component unit struct, and — with the `serde` feature
/// on — give it a `#[serde(tag = "type")]` envelope whose tag is *required*.
///
/// The requirement is the whole point, and it is not what `#[serde(tag = "type")]` alone gives you:
/// that attribute only *writes* the tag, and ignores it entirely on the way in. The `Def` remote
/// plus the `Deserializer` shim below are what make a missing `"type"` an error rather than a
/// silently-accepted bare struct, and the unit-struct arm does the same job with a `Helper` whose
/// only field is the tag.
///
/// It matters because every wrapper enum in `tk-convert` has an *untagged* legacy fallback: a
/// variant that is lenient about the tag will happily claim a tag-less object that should have been
/// rejected. `pre_tokenizer_deserialization_no_type` and `decoder_serialization_no_decode` are the
/// tests that catch it.
///
/// Call sites reach it through `#[cfg_attr(feature = "serde", macro_rules_attribute(impl_serde_type!))]`,
/// so with the feature off the struct is declared and nothing else — no serde attributes are
/// applied to it, and none may appear inside it either. A field that needs one spells it
/// `#[cfg_attr(feature = "serde", serde(default = "…"))]`.
#[cfg(feature = "serde")]
#[macro_export]
macro_rules! impl_serde_type{
    (
     $(#[$meta:meta])*
     $vis:vis struct $struct_name:ident {
        $(
        $(#[$field_meta:meta])*
        $field_vis:vis $field_name:ident : $field_type:ty
        ),*$(,)+
    }
    ) => {
        paste::paste!{
            $(#[$meta])*
            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(tag = "type", from = $struct_name "Deserializer")]
            $vis struct $struct_name{
                $(
                    $(#[$field_meta])*
                    $field_vis $field_name : $field_type,
                )*
            }

            // Everything below exists only to give serde something to drive: the `Def` remote, the
            // type-tag enum, the `Deserializer` shim and the `From` that unwraps it.
            #[doc(hidden)]
            $(#[$meta])*
            #[derive(serde::Deserialize)]
            #[serde(tag = "type", remote = $struct_name "")]
            struct [<$struct_name Def>]{
                $(
                    $(#[$field_meta])*
                    $field_vis $field_name : $field_type,
                )*
            }

            #[doc(hidden)]
            #[derive(serde::Deserialize)]
            enum [<$struct_name Type>] {
                $struct_name,
            }

            #[doc(hidden)]
            #[derive(serde::Deserialize)]
            struct [<$struct_name Deserializer>] {
                #[allow(dead_code)]
                r#type: [<$struct_name Type>],
                #[serde(flatten, with = $struct_name "Def")]
                r#struct: $struct_name,
            }

            #[doc(hidden)]
            impl std::convert::From<[<$struct_name Deserializer>]> for $struct_name {
                fn from(v: [<$struct_name Deserializer>]) -> Self {
                    v.r#struct
                }
            }
        }
    };
    (
     $(#[$meta:meta])*
     $vis:vis struct $struct_name:ident;
    ) => {
        paste::paste!{
            $(#[$meta])*
            $vis struct $struct_name;

            impl serde::Serialize for $struct_name {
                fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
                where
                    S: serde::ser::Serializer,
                {
                    let helper = [<$struct_name Helper>]{r#type: [<$struct_name Type>]::$struct_name};
                    helper.serialize(serializer)
                }
            }

            impl<'de> serde::Deserialize<'de> for $struct_name {
                fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
                where
                    D: serde::Deserializer<'de>,
                {
                    let _helper = [<$struct_name Helper>]::deserialize(deserializer)?;
                    Ok($struct_name)
                }
            }

            #[doc(hidden)]
            #[derive(serde::Serialize, serde::Deserialize)]
            enum [<$struct_name Type>] {
                $struct_name,
            }

            #[doc(hidden)]
            #[derive(serde::Serialize, serde::Deserialize)]
            struct [<$struct_name Helper>] {
                #[allow(dead_code)]
                r#type: [<$struct_name Type>],
            }
        }
    }
}

/// The declaration-only half of [`impl_serde_type`], for a build with the `serde` feature off.
///
/// Call sites apply the macro through `cfg_attr`, so with the feature off it is never invoked at
/// all — but it stays defined and `macro_export`ed so that the *set* of components spelled this way
/// is one grep either way, and so a downstream crate naming the macro does not break when the
/// feature flips.
#[cfg(not(feature = "serde"))]
#[macro_export]
macro_rules! impl_serde_type{
    (
     $(#[$meta:meta])*
     $vis:vis struct $struct_name:ident {
        $(
        $(#[$field_meta:meta])*
        $field_vis:vis $field_name:ident : $field_type:ty
        ),*$(,)+
    }
    ) => {
        $(#[$meta])*
        $vis struct $struct_name{
            $(
                $(#[$field_meta])*
                $field_vis $field_name : $field_type,
            )*
        }
    };
    (
     $(#[$meta:meta])*
     $vis:vis struct $struct_name:ident;
    ) => {
        $(#[$meta])*
        $vis struct $struct_name;
    }
}

// Re-export macro_rules_attribute
pub use macro_rules_attribute::macro_rules_attribute;
