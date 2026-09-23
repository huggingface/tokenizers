//! Various "bucket functions" are implemented here.
//!
//! These functions map a uniform `u64` hash to a new `u64`, to skew the distribution of bucket sizes.
//!
//! By default, `CubicEps` is used. `Linear` is simplest and can be faster at the cost of requiring more space.
//! The remaining ones are only kept for benchmarking.

use mem_dbg::MemSize;

use crate::util::mul_high;
use std::fmt::Debug;

pub trait BucketFn: Clone + Copy + Sync + Debug {
    const LINEAR: bool = false;
    const B_OUTPUT: bool = false;
    fn set_buckets_per_part(&mut self, _b: u64) {}
    fn call(&self, x: u64) -> u64;
}

/// The function simply returns `x` itself.
#[derive(Clone, Copy, Debug, MemSize, Default)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct Linear;

impl BucketFn for Linear {
    const LINEAR: bool = true;
    fn call(&self, x: u64) -> u64 {
        x
    }
}

/// A 2-piece-wise linear function, as used in FCH and PTHash.
///
/// |              .
/// |             .
/// |         ....---< gamma
/// |    .....   |
/// |....        |
/// +------------^--
///              beta
///
/// line1: y = x * (gamma / beta)
///                ~~~ slope1 ~~~
/// line2: y = x * ((1 - gamma) / (1 - beta)) + (gamma - beta) / (1 - beta)
///                ~~~~~~~~~ slope2 ~~~~~~~~~   ~~~~~~~~~~ offset ~~~~~~~~~
#[derive(Clone, Copy, Debug, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct Skewed {
    beta_f: f64,
    gamma_f: f64,
    /// buckets per part
    b: u64,
    beta: u64,
    slope1: u64,
    slope2: u64,
    neg_offset: u64,
}

impl Default for Skewed {
    fn default() -> Self {
        Skewed::new(0.6, 0.3)
    }
}

impl Skewed {
    // Map the first beta% of hashes to the first gamma% of buckets.
    pub fn new(beta: f64, gamma: f64) -> Self {
        assert!(
            beta > gamma,
            "Beta={beta} must be larger than gamma={gamma}"
        );
        Self {
            beta_f: beta,
            gamma_f: gamma,
            b: 0,
            beta: 0,
            slope1: 0,
            slope2: 0,
            neg_offset: 0,
        }
    }
}

impl BucketFn for Skewed {
    const B_OUTPUT: bool = true;
    fn set_buckets_per_part(&mut self, b: u64) {
        let beta = self.beta_f;
        let gamma = self.gamma_f;
        self.b = b;
        let as_u64 = |x: f64| (x * u64::MAX as f64) as u64;
        self.slope1 = mul_high(as_u64(gamma / beta), self.b);
        self.slope2 = mul_high(as_u64((1. - gamma) / (1. - beta) / 8.), self.b << 3);
        self.neg_offset = mul_high(as_u64((beta - gamma) / (1. - beta) / 8.), self.b << 3);
        self.beta = as_u64(beta);
    }
    fn call(&self, x: u64) -> u64 {
        // NOTE: There is a lot of MOV/CMOV going on here.
        let is_large = x >= self.beta;
        let slope = if is_large { self.slope2 } else { self.slope1 };
        mul_high(x, slope) - is_large as u64 * self.neg_offset
        // debug_assert!(!is_large || self.p2 <= b, "p2 {} <= b {}", self.p2, b);
        // debug_assert!(!is_large || b < self.b, "b {} < p2 {}", b, self.b);
        // debug_assert!(is_large || b < self.p2, "b {} < p2 {}", b, self.p2);
    }
}

/// The optimal bucket function of PHOBIC, with a variable `eps`.
#[derive(Clone, Copy, Debug, MemSize)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct Optimal {
    pub eps: f64,
}

impl BucketFn for Optimal {
    fn call(&self, x: u64) -> u64 {
        let p32 = (1u64 << 32) as f64;
        let p64 = p32 * p32;
        let p64inv = 1. / p64;
        let x = (x as f64) * p64inv;
        let y = x + (1. - self.eps) * (1. - x) * (1. - x).ln();

        (y * p64) as u64
    }
}

/// `x*x`
#[derive(Clone, Copy, Debug, MemSize, Default)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct Square;

impl BucketFn for Square {
    fn call(&self, x: u64) -> u64 {
        mul_high(x, x)
    }
}

/// `x*x * 255/256 + x/256`
#[derive(Clone, Copy, Debug, MemSize, Default)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct SquareEps;

impl BucketFn for SquareEps {
    fn call(&self, x: u64) -> u64 {
        mul_high(x, x) / 256 * 255 + x / 256
    }
}

/// `x * x * (1 + x)/2`
#[derive(Clone, Copy, Debug, MemSize, Default)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct Cubic;

impl BucketFn for Cubic {
    fn call(&self, x: u64) -> u64 {
        // x * x * (1 + x)/2
        mul_high(mul_high(x, x), (x >> 1) | (1 << 63))
    }
}

/// `x * x * (1 + x)/2 * 255/256 + x/256`
#[derive(Clone, Copy, Debug, MemSize, Default)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::prelude::Epserde))]
#[cfg_attr(feature = "epserde", epserde(deep_copy))]
pub struct CubicEps;

impl BucketFn for CubicEps {
    fn call(&self, x: u64) -> u64 {
        // x * x * (1 + x)/2
        mul_high(mul_high(x, x), (x >> 1) | (1 << 63)) / 256 * 255 + x / 256
    }
}

#[cfg(test)]
mod test {
    use crate::bucket_fn::BucketFn;

    #[test]
    fn test_skewed() {
        use super::Skewed;
        let mut skewed = Skewed::new(0.6, 0.3);
        skewed.set_buckets_per_part(1000000000);

        let mut last_y = 0;
        let n = 100;
        for i in 0..100 {
            let x = u64::MAX / n * i;
            let y = skewed.call(x);
            assert!(y >= last_y);
            last_y = y;
        }
    }
}
