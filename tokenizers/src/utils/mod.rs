pub(crate) mod cache;
#[cfg(feature = "http")]
pub(crate) mod from_pretrained;
pub mod iter;
pub mod padding;
pub mod parallelism;
pub(crate) mod progress;
pub mod truncation;

use serde::{Serialize, Serializer};
use std::collections::{BTreeMap, HashMap};

pub(crate) fn ordered_map<S, K, V>(
    value: &HashMap<K, V>,
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

macro_rules! impl_serde_unit_struct (
    ($visitor:ident, $self_ty:tt) => {
        impl serde::Serialize for $self_ty {
            fn serialize<S>(&self, serializer: S)  -> std::result::Result<S::Ok, S::Error> where
                S: serde::ser::Serializer {
                    use serde::ser::SerializeStruct;
                    let self_ty_str = stringify!($self_ty);
                    let mut m = serializer.serialize_struct(self_ty_str,1)?;
                    m.serialize_field("type", self_ty_str)?;
                    m.end()
            }
        }

        impl<'de> serde::Deserialize<'de> for $self_ty {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error> where
                D: serde::de::Deserializer<'de> {
                deserializer.deserialize_map($visitor)
            }
        }

        struct $visitor;
        impl<'de> serde::de::Visitor<'de> for $visitor {
            type Value = $self_ty;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, stringify!($self_ty))
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error> where
                A: serde::de::MapAccess<'de>, {
                let self_ty_str = stringify!($self_ty);
                let maybe_type = map.next_entry::<String, String>()?;
                let maybe_type_str = maybe_type.as_ref().map(|(k, v)| (k.as_str(), v.as_str()));
                match maybe_type_str {
                    Some(("type", stringify!($self_ty))) => Ok($self_ty),
                    Some((_, ty)) => Err(serde::de::Error::custom(&format!("Expected {}, got {}", self_ty_str, ty))),
                    None => Err(serde::de::Error::custom(&format!("Expected type : {}", self_ty_str)))
                }
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
        use paste::paste;

        paste!{
            $(#[$meta])*
            #[derive(Serialize, Deserialize)]
            #[serde(tag = "type", from = $struct_name "Deserilaizer")]
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
            struct [<$struct_name Deserilaizer>] {
                #[allow(dead_code)]
                r#type: [<$struct_name Type>],
                #[serde(flatten, with = $struct_name "Def")]
                r#struct: $struct_name,
            }

            #[doc(hidden)]
            impl std::convert::From<[<$struct_name Deserilaizer>]> for $struct_name {
                fn from(v: [<$struct_name Deserilaizer>]) -> Self {
                    v.r#struct
                }
            }
        }
    };
    (
     $(#[$meta:meta])*
     $vis:vis struct $struct_name:ident;
    ) => {
        use paste::paste;

        paste!{
            $(#[$meta])*
            $vis struct $struct_name;

            impl serde::Serialize for $struct_name {
                fn serialize<S>(&self, serializer: S)  -> std::result::Result<S::Ok, S::Error> where
                    S: serde::ser::Serializer {
                        use serde::ser::SerializeStruct;
                        let struct_name_str = stringify!($struct_name);
                        let mut m = serializer.serialize_struct(struct_name_str,1)?;
                        m.serialize_field("type", struct_name_str)?;
                        m.end()
                }
            }

            impl<'de> serde::Deserialize<'de> for $struct_name {
                fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error> where
                    D: serde::de::Deserializer<'de> {
                    deserializer.deserialize_map([<$struct_name Visitor>])
                }
            }

            struct [<$struct_name Visitor>];
            impl<'de> serde::de::Visitor<'de> for [<$struct_name Visitor>] {
                type Value = $struct_name;
                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(formatter, stringify!($struct_name))
                }

                fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error> where
                    A: serde::de::MapAccess<'de>, {
                    let struct_name_str = stringify!($struct_name);
                    let maybe_type = map.next_entry::<String, String>()?;
                    let maybe_type_str = maybe_type.as_ref().map(|(k, v)| (k.as_str(), v.as_str()));
                    match maybe_type_str {
                        Some(("type", stringify!($struct_name))) => Ok($struct_name),
                        Some((_, ty)) => Err(serde::de::Error::custom(&format!("Expected {}, got {}", struct_name_str, ty))),
                        None => Err(serde::de::Error::custom(&format!("Expected type : {}", struct_name_str)))
                    }
                }
            }
        }
    }
}

// Re-export macro_rules_attribute
pub use proc_macros::macro_rules_attribute;
