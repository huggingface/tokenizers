pub mod iter;
pub mod padding;
pub mod parallelism;
pub mod truncation;

#[macro_use]
macro_rules! impl_enum_from(
    ($from_ty:ty, $enum:ty, $variant:ident) => {
        impl From<$from_ty> for $enum {
            fn from(from: $from_ty) -> Self {
                <$enum>::$variant(from)
            }
        }
    }
);
