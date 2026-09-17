pub(crate) mod cache;
#[cfg(feature = "http")]
pub(crate) mod from_pretrained;

#[cfg(all(
    feature = "fancy-regex",
    not(feature = "onig"),
    not(feature = "rusty-expressions")
))]
mod fancy;
#[cfg(all(
    feature = "fancy-regex",
    not(feature = "onig"),
    not(feature = "rusty-expressions")
))]
pub use fancy::SysRegex;
// `rusty_expressions` is Oniguruma reimplemented in pure Rust, so it gives the
// same engine semantics as `onig` with no C in the build. It wins over both
// when enabled.
#[cfg(feature = "rusty-expressions")]
mod rusty;
#[cfg(feature = "rusty-expressions")]
pub use crate::utils::rusty::SysRegex;
#[cfg(all(feature = "onig", not(feature = "rusty-expressions")))]
mod onig;
#[cfg(all(feature = "onig", not(feature = "rusty-expressions")))]
pub use crate::utils::onig::SysRegex;

/// Which regex engine this build actually compiled in.
///
/// The backend features are resolved by precedence, and Cargo features are
/// additive: anything anywhere in the dependency graph can switch
/// `rusty-expressions` on, and it then wins over `onig` and `fancy-regex` for
/// every crate in that graph. That is a silent change of engine for someone
/// who asked for a different one, and the regex engine decides how text is
/// split -- so it is worth being able to ask.
///
/// A `compile_error!` would be the loud alternative, but it would also break
/// any build where a transitive dependency turned the feature on, which is a
/// legal thing for a dependency to do. So the answer is reported rather than
/// enforced: assert on it if your build cares.
///
/// ```
/// // Pin the engine your tokenizer outputs were produced with, in a build
/// // that selects it -- this doctest runs under whatever features are on.
/// let backend = tokenizers::utils::REGEX_BACKEND;
/// assert!(matches!(backend, "rusty_expressions" | "onig" | "fancy-regex"));
/// ```
pub const REGEX_BACKEND: &str = if cfg!(feature = "rusty-expressions") {
    "rusty_expressions"
} else if cfg!(feature = "onig") {
    "onig"
} else {
    "fancy-regex"
};

/// True when `onig` was requested but another backend took precedence.
///
/// In this state libonig is still compiled and linked -- the C toolchain cost
/// is paid -- and then never called. Worth asserting against in CI if you
/// meant to have dropped one or the other.
pub const REGEX_BACKEND_OVERRODE_ONIG: bool =
    cfg!(feature = "onig") && cfg!(feature = "rusty-expressions");

#[cfg(not(any(
    feature = "onig",
    feature = "fancy-regex",
    feature = "rusty-expressions"
)))]
compile_error!(
    "One of the `onig`, `fancy-regex`, or `rusty-expressions` features must be enabled"
);

pub mod iter;
pub mod padding;
pub mod parallelism;
pub(crate) mod progress;
pub mod truncation;

// Re-export ProgressFormat for public API
pub use progress::ProgressFormat;

use ahash::AHashMap;
use serde::{Serialize, Serializer};
use std::collections::BTreeMap;

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

#[cfg(test)]
mod backend_tests {
    /// The reported backend must be the one actually compiled in.
    ///
    /// Cheap insurance that `REGEX_BACKEND` cannot drift away from the `cfg`
    /// chain above it and start reporting an engine this build does not have.
    #[test]
    fn reported_backend_matches_the_compiled_one() {
        let re = super::SysRegex::new("a+").expect("compiles");
        assert!(re.find_iter("baaad").next().is_some());
        let expected = if cfg!(feature = "rusty-expressions") {
            "rusty_expressions"
        } else if cfg!(feature = "onig") {
            "onig"
        } else {
            "fancy-regex"
        };
        assert_eq!(super::REGEX_BACKEND, expected);
        assert_eq!(
            super::REGEX_BACKEND_OVERRODE_ONIG,
            cfg!(feature = "onig") && cfg!(feature = "rusty-expressions")
        );
    }
}
