use mem_dbg::MemSize;

use crate::reduce::Reduce;

// Multiply a u128 by u64 and return the upper 64 bits of the result.
// ((lowbits * d as u128) >> 128) as u64
#[allow(unused)]
fn mul128_u64(lowbits: u128, d: u64) -> u64 {
    let bot_half = ((lowbits & u64::MAX as u128) * d as u128) >> 64; // Won't overflow
    let top_half = (lowbits >> 64) * d as u128;
    let both_halves = bot_half + top_half; // Both halves are already shifted down by 64
    (both_halves >> 64) as u64
}

/// FastMod64
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
#[allow(unused)]
pub struct FM64 {
    d: u64,
    m: u128,
}
impl Reduce for FM64 {
    fn new(d: usize) -> Self {
        Self {
            d: d as u64,
            m: (u128::MAX / d as u128).wrapping_add(1),
        }
    }
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m.wrapping_mul(h as u128);
        mul128_u64(lowbits, self.d) as usize
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct FM32 {
    d: u64,
    m: u64,
}
impl Reduce for FM32 {
    fn new(d: usize) -> Self {
        // assert!(d <= u32::MAX as usize);
        Self {
            d: d as u64,
            m: (u64::MAX / d as u64).wrapping_add(1),
        }
    }
    // NOTE: h must be < 2^32 for correct modulo results!.
    // Otherwise, a 'random' integer up to d is returned.
    fn reduce(self, h: u64) -> usize {
        let lowbits = self.m.wrapping_mul(h);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
}

#[test]
fn test_fastmod64() {
    #[cfg(debug_assertions)]
    let cnt = 100;
    #[cfg(not(debug_assertions))]
    let cnt = 10000;

    for _ in 1..cnt {
        let d = rand::random::<u64>() as usize + 1;
        let fm = FM64::new(d);
        for _ in 0..cnt {
            let x = rand::random::<u64>() as u64;
            assert_eq!(fm.reduce(x), x as usize % d, "failure for d = {d}, x = {x}",);
        }
    }
}

#[test]
fn test_fastmod32_equals_modulo() {
    #[cfg(debug_assertions)]
    let cnt = 100;
    #[cfg(not(debug_assertions))]
    let cnt = 10000;

    for _ in 1..cnt {
        let d = rand::random::<u32>() as usize + 1;
        let fm = FM32::new(d);
        for _ in 0..cnt {
            let x = rand::random::<u32>() as u64;
            assert_eq!(fm.reduce(x), x as usize % d, "failure for d = {d}, x = {x}",);
        }
    }
}

#[test]
fn test_fastmod32_doesnt_overflow() {
    #[cfg(debug_assertions)]
    let cnt = 100;
    #[cfg(not(debug_assertions))]
    let cnt = 10000;

    for _ in 1..cnt {
        let d = rand::random::<u64>().saturating_add(1) as usize;
        let fm = FM32::new(d);
        for _ in 0..cnt {
            let x = rand::random::<u64>();
            assert!(fm.reduce(x) < d, "failure for d = {d}, x = {x}",);
        }
    }
}
