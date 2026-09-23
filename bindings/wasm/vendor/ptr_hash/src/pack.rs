//! The `Packed` and `MutPacked` traits are used for the underlying storage of
//! the remap vector.
//!
//! This is implemented for `Vec<u8|u16|u32|u64>`, `CachelineEfVec`, and `EliasFano` from `sucds`.
//! `Packed` is also implemented for respective non-owning (slice) types to support epserde.

#[cfg(feature = "elias-fano")]
use sucds::mii_sequences::EliasFanoBuilder;

#[cfg(feature = "cacheline-ef")]
use cacheline_ef::{CachelineEf, CachelineEfVec};

/// A trait for backing storage types.
pub trait Packed: Sync {
    /// This uses get_unchecked internally, so you must ensure that index is within bounds.
    fn index(&self, index: usize) -> u64;
    /// Prefetch the element at the given index.
    fn prefetch(&self, _index: usize) {}
    /// Size in bytes.
    fn size_in_bytes(&self) -> usize;
}

/// An extension of Packed that can be used during construction.
pub trait MutPacked: Packed + Sized {
    fn default() -> Self;
    fn try_new(vals: Vec<u64>) -> Option<Self>;
    fn name() -> String;
}

macro_rules! vec_impl {
    ($t:ty) => {
        impl MutPacked for Vec<$t> {
            fn default() -> Self {
                Default::default()
            }
            fn try_new(vals: Vec<u64>) -> Option<Self> {
                Some(
                    vals.into_iter()
                        .map(|x| {
                            x.try_into()
                                .expect(&format!("Value {x} is larger than backing type can hold."))
                        })
                        .collect(),
                )
            }
            fn name() -> String {
                stringify!(Vec<$t>).to_string()
            }
        }
        impl Packed for Vec<$t> {
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                prefetch_index::prefetch_index(self, index);
            }
            fn size_in_bytes(&self) -> usize {
                std::mem::size_of_val(self.as_slice())
            }
        }
    };
}

vec_impl!(u8);
vec_impl!(u16);
vec_impl!(u32);
vec_impl!(u64);

macro_rules! slice_impl {
    ($t:ty) => {
        impl Packed for [$t] {
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                prefetch_index::prefetch_index(self, index);
            }
            fn size_in_bytes(&self) -> usize {
                std::mem::size_of_val(self)
            }
        }
        impl Packed for &[$t] {
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                prefetch_index::prefetch_index(self, index);
            }
            fn size_in_bytes(&self) -> usize {
                std::mem::size_of_val(*self)
            }
        }
        impl Packed for Box<[$t]> {
            fn index(&self, index: usize) -> u64 {
                unsafe { (*self.get_unchecked(index)) as u64 }
            }
            fn prefetch(&self, index: usize) {
                prefetch_index::prefetch_index(self, index);
            }
            fn size_in_bytes(&self) -> usize {
                std::mem::size_of_val(&*self)
            }
        }
    };
}

slice_impl!(u8);
slice_impl!(u16);
slice_impl!(u32);
slice_impl!(u64);

#[cfg(feature = "cacheline-ef")]
impl MutPacked for CachelineEfVec<Vec<CachelineEf>> {
    fn default() -> Self {
        Default::default()
    }
    fn try_new(vals: Vec<u64>) -> Option<Self> {
        Self::try_new(&vals)
    }
    fn name() -> String {
        "CacheLineEF".to_string()
    }
}

#[cfg(feature = "cacheline-ef")]
impl<T: AsRef<[CachelineEf]> + Sync> Packed for CachelineEfVec<T> {
    fn index(&self, index: usize) -> u64 {
        unsafe { self.index_unchecked(index) }
    }
    fn prefetch(&self, index: usize) {
        self.prefetch(index)
    }
    fn size_in_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

/// Wrapper around the Sucds implementation.
#[cfg(feature = "elias-fano")]
pub struct EliasFano(sucds::mii_sequences::EliasFano);

#[cfg(feature = "elias-fano")]
impl MutPacked for EliasFano {
    fn default() -> Self {
        EliasFano(Default::default())
    }

    fn try_new(vals: Vec<u64>) -> Option<Self> {
        if vals.is_empty() {
            Some(Self::default())
        } else {
            let mut builder =
                EliasFanoBuilder::new(*vals.last().unwrap() as usize + 1, vals.len()).unwrap();
            builder.extend(vals.iter().map(|&x| x as usize)).unwrap();
            Some(EliasFano(builder.build()))
        }
    }
    fn name() -> String {
        "EF".to_string()
    }
}

#[cfg(feature = "elias-fano")]
impl Packed for EliasFano {
    fn index(&self, index: usize) -> u64 {
        self.0.select(index as _).unwrap() as u64
    }

    fn size_in_bytes(&self) -> usize {
        sucds::Serializable::size_in_bytes(&self.0)
    }
}
