pub mod cache;
pub mod iter;
pub mod padding;
pub mod parallelism;
pub mod progress;
pub mod truncation;

use serde::{Serialize, Serializer};
use std::collections::{BTreeMap, HashMap};

pub fn ordered_map<S, K, V>(
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

#[macro_use]
macro_rules! impl_enum_from (
    ($from_ty:ty, $enum:ty, $variant:ident) => {
        impl From<$from_ty> for $enum {
            fn from(from: $from_ty) -> Self {
                <$enum>::$variant(from)
            }
        }
    }
);

#[macro_use]
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
