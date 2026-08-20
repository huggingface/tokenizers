//! The two macros the moved code is written against.
//!
//! Both exist in `tk-encode` too, and deliberately are not reused from there: over there serde is a
//! feature, so `tk-encode`'s `impl_serde_type!` is defined twice under a `#[cfg]` and its call sites
//! reach it through `cfg_attr`. A `macro_rules!` expands in the *calling* crate, so importing that
//! one would read the cfg against *this* crate's features — and this crate has no `serde` feature,
//! being the serde layer itself. Rather than declare a feature that is always on, these copies have
//! no gates.

/// `impl From<$from_ty> for $enum` for one variant of a wrapper enum.
macro_rules! impl_enum_from (
    ($from_ty:ty, $enum:ty, $variant:ident) => {
        impl From<$from_ty> for $enum {
            fn from(from: $from_ty) -> Self {
                <$enum>::$variant(from)
            }
        }
    }
);
pub(crate) use impl_enum_from;

/// `Serialize`/`Deserialize` with a `#[serde(tag = "type")]` envelope whose tag is *required*.
///
/// The requirement is the whole point, and it is not what `#[serde(tag = "type")]` alone gives you:
/// the `Def` remote plus the `Deserializer` shim make a missing `"type"` an error rather than a
/// silently-accepted bare struct. The wrapper enums' untagged legacy fallbacks depend on that — a
/// `Sequence` with no tag must fail its own deserializer so the fallback can try the next variant.
macro_rules! impl_serde_type {
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
}
pub(crate) use impl_serde_type;
