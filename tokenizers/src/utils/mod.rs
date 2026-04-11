pub(crate) mod cache;
#[cfg(feature = "http")]
pub(crate) mod from_pretrained;

#[cfg(all(feature = "fancy-regex", not(feature = "onig")))]
mod fancy;
#[cfg(all(feature = "fancy-regex", not(feature = "onig")))]
pub use fancy::SysRegex;
#[cfg(feature = "onig")]
mod onig;
#[cfg(feature = "onig")]
pub use crate::utils::onig::SysRegex;

#[cfg(not(any(feature = "onig", feature = "fancy-regex")))]
compile_error!("One of the `onig`, or `fancy-regex` features must be enabled");

pub mod iter;
pub mod padding;
pub mod parallelism;
pub(crate) mod progress;
pub mod truncation;

// Re-export ProgressFormat for public API
pub use progress::ProgressFormat;

use serde::{Serialize, Serializer};
use std::collections::BTreeMap;

/// Fast hash map using foldhash. Drop-in replacement for ahash::AHashMap.
pub type AHashMap<K, V> = std::collections::HashMap<K, V, foldhash::fast::FixedState>;
/// Fast hash set using foldhash. Drop-in replacement for ahash::AHashSet.
pub type AHashSet<K> = std::collections::HashSet<K, foldhash::fast::FixedState>;

// Re-export extension traits so AHashMap::new() and AHashSet::new() work.
#[allow(unused_imports)]
pub use foldhash::HashMapExt;
#[allow(unused_imports)]
pub use foldhash::HashSetExt;

pub(crate) fn ordered_map<S, K, V>(
    value: &AHashMap<K, V>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Serialize + std::cmp::Ord,
    V: Serialize,
{
    let ordered: BTreeMap<_, _> = value.iter().collect();
    ordered.serialize(serializer)
}

macro_rules! impl_enum_from (
    ($from_ty:ty, $enum:ty, $variant:ident) => {
        impl From<$from_ty> for $enum {
            fn from(from: $from_ty) -> Self {
                <$enum>::$variant(from)
            }
        }
    }
);

/// Implement `serde::{Serialize, Serializer}` with `#[serde(tag = "type")]` attribute for a given struct.
/// Panic when a json string being deserilized misses field `type`.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate tokenizers;
/// use serde::{Serialize, Deserialize};
///
/// fn main() {
///    impl_serde_type!{
///        #[derive(Debug)]
///        struct Point {
///            x: i32,
///            #[serde(default = "default_y")]
///            y: i32,
///        }
///    }
///    fn default_y() -> i32 {
///        5
///    }
///
///    let point = Point { x: 1, y: 2 };
///    let serialized_s = r#"{"type":"Point","x":1,"y":2}"#;
///    assert_eq!(serde_json::to_string(&point).unwrap(), serialized_s);
/// }
/// ```
///
/// ```should_panic
/// # #[macro_use] extern crate tokenizers;
/// use serde::{Serialize, Deserialize};
///
/// fn main() {
///    impl_serde_type!{
///        #[derive(Debug)]
///        struct Point1D {
///            x: i32,
///        }
///    }
///
///    let serialized_s = r#"{"x":1}"#;
///    let deserialized: Point1D = serde_json::from_str(serialized_s).unwrap();
/// }
/// ```
///
/// # Examples (unit structs)
///
/// ```
/// # #[macro_use] extern crate tokenizers;
/// use serde::{Serialize, Deserialize};
///
/// fn main() {
///    impl_serde_type!{
///        struct Unit;
///    }
///
///    let unit = Unit;
///    let serialized_s = r#"{"type":"Unit"}"#;
///    assert_eq!(serde_json::to_string(&unit).unwrap(), serialized_s);
/// }
/// ```
///
/// ```should_panic
/// # #[macro_use] extern crate tokenizers;
/// use serde::{Serialize, Deserialize};
///
/// fn main() {
///    impl_serde_type!{
///        struct Unit;
///    }
///
///    let serialized_s = r#"{"some_field":1}"#;
///    let deserialized: Unit = serde_json::from_str(serialized_s).unwrap();
/// }
/// ```
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
            #[derive(Serialize, Deserialize)]
            #[serde(tag = "type", from = $struct_name "Deserializer")]
            $vis struct $struct_name{
                $(
                    $(#[$field_meta])*
                    $field_vis $field_name : $field_type,
                )*
            }

            #[doc(hidden)]
            $(#[$meta])*
            #[derive(Deserialize)]
            #[serde(tag = "type", remote = $struct_name "")]
            struct [<$struct_name Def>]{
                $(
                    $(#[$field_meta])*
                    $field_vis $field_name : $field_type,
                )*
            }

            #[doc(hidden)]
            #[derive(Deserialize)]
            enum [<$struct_name Type>] {
                $struct_name,
            }

            #[doc(hidden)]
            #[derive(Deserialize)]
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
                fn serialize<S>(&self, serializer: S)  -> std::result::Result<S::Ok, S::Error> where
                    S: serde::ser::Serializer {
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

            #[derive(serde::Serialize, serde::Deserialize)]
            enum [<$struct_name Type>] {
                $struct_name,
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            struct [<$struct_name Helper>] {
                #[allow(dead_code)]
                r#type: [<$struct_name Type>],
            }
        }
    }
}

// Re-export macro_rules_attribute
pub use macro_rules_attribute::macro_rules_attribute;

/// Deserialize from a `serde_json::Value` via string round-trip.
/// This avoids monomorphizing `serde_json::from_value<T>` per type — the
/// `from_str` path shares a single `Deserializer<SliceRead>` across all types,
/// saving ~66 KB of binary size compared to `serde_json::from_value`.
pub(crate) fn from_value_via_str<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
) -> serde_json::Result<T> {
    let s = value.to_string();
    serde_json::from_str(&s)
}

/// Escape special regex metacharacters in a string so it can be used as a literal pattern.
/// This replaces the dependency on `regex::escape`.
pub fn regex_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len() + 4);
    for c in text.chars() {
        match c {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
            | '#' | '&' | '-' | '~' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

/// Check if a character is a "word" character (equivalent to `\w` in Unicode-aware regex).
/// Matches `[a-zA-Z0-9_]`, Unicode alphabetic/numeric characters, and Unicode combining marks
/// (category M), matching Perl/PCRE `\w` with Unicode enabled.
pub(crate) fn is_word_char(c: char) -> bool {
    use unicode_categories::UnicodeCategories;
    c == '_' || c.is_alphanumeric() || c.is_mark()
}
