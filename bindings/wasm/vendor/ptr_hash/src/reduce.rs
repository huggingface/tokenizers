use mem_dbg::MemSize;

use crate::{
    hash::{self},
    util::mul_high,
};

pub trait Reduce: Copy + Sync + std::fmt::Debug {
    /// Reduce into the range [0, d).
    fn new(d: usize) -> Self;
    /// Reduce a (uniform random 64 bit) number into the range [0, d).
    fn reduce(self, h: u64) -> usize;
    /// Reduce a (uniform random 64 bit) number into the range [0, d),
    /// and also return a remainder that can be used for further reductions.
    fn reduce_with_remainder(self, _h: u64) -> (usize, u64) {
        unimplemented!();
    }
}

/// FastReduce64
/// Taken from https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
/// NOTE: This only uses the lg(n) high-order bits of entropy from the hash.
#[derive(Copy, Clone, Debug, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct FastReduce {
    d: u64,
}
impl Reduce for FastReduce {
    fn new(d: usize) -> Self {
        Self { d: d as u64 }
    }
    fn reduce(self, h: u64) -> usize {
        mul_high(self.d, h) as usize
    }
    fn reduce_with_remainder(self, h: u64) -> (usize, u64) {
        let r = self.d as u128 * h as u128;
        ((r >> 64) as usize, r as u64)
    }
}

/// Multiply-Reduce 64
/// Multiply by mixing constant C and take the required number of bits.
/// Only works when the modulus is a power of 2.
#[derive(Copy, Clone, Debug, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
#[allow(unused)]
pub struct MulReduce {
    mask: u64,
}
impl Reduce for MulReduce {
    fn new(d: usize) -> Self {
        assert!(d.is_power_of_two(), "{d} is not a power of 2");
        Self {
            mask: (d - 1) as u64,
        }
    }
    fn reduce(self, h: u64) -> usize {
        (mul_high(hash::C, h) & self.mask) as usize
    }
}
