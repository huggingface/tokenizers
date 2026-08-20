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

// `ordered_map`, the `#[serde(serialize_with = ...)]` helper that gave the `TemplateProcessing`
// special-token table a deterministic key order, moved to `tk-convert`'s `mirror` module along with
// the processor that was its only caller. Nothing in this crate serializes anything any more.

/// Declare a pipeline component struct, or a component unit struct.
///
/// The name is historical. This macro used to *also* emit `Serialize`/`Deserialize` and the
/// `Def`/`Type`/`Deserializer` plumbing that made the `"type"` tag mandatory rather than merely
/// emitted. None of that is here any more, because this crate links no serde at all -- so what is
/// left is the declaration.
///
/// The serde half lives in `tk-convert`, twice over, and deliberately not shared with this file:
///
/// * `tk_convert::macros::impl_serde_type` is this same macro *with* its serde arms, for the config
///   layer's own types -- the four `Sequence` components;
/// * each component's on-disk shape is a mirror next to the wrapper that holds it
///   (`decoders::mirror`, `normalizers::mirror`, `pre_tokenizers::mirror`, `processors::mirror`),
///   because a foreign crate cannot add a `Deserialize` impl to a type declared here.
///
/// Whether a mirror *requires* the `"type"` tag is decided per type, and it matters: a wrapper's
/// legacy fallback is an untagged enum, so a lenient variant will claim a tag-less object that
/// should have been rejected. Every type spelled with this macro used to get a required tag, so
/// every one of their mirrors reproduces that.
///
/// Kept as a macro rather than expanded away at its call sites so the *set* of components spelled
/// this way stays greppable, and so that a future build wanting tags back has one place to put them.
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
