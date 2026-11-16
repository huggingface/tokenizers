//! The perfect hash behind the vocabulary, the BPE merge table and the other fixed lookup tables.
//!
//! A perfect hash is built once from a fixed set of keys and sends each of them to its own slot, so
//! there are no collisions to resolve. A key outside the set still lands on some slot, so a table
//! keeps the key (or something derived from it) in the slot and compares it on lookup.
//!
//! [`Mphf`] takes keys that are already hashed. Integer keys go through [`mix`] first, or
//! [`splitmix64`] for dense runs such as token ids.

use crate::Result;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};

/// The one perfect-hash type in the crate, over ptr_hash's [`FastPtrHash`].
///
/// `FastPtrHash` is generic, so each other instantiation would add another copy of its code to the
/// binary.
#[derive(Clone)]
pub(crate) struct Mphf(FastPtrHash<NoHash, u64>);

impl Mphf {
    /// Builds the perfect hash of `keys`.
    ///
    /// Fails when ptr_hash finds no placement for the keys, for example when two of them are equal.
    pub(crate) fn try_new(keys: &[u64]) -> Result<Self> {
        let default = PtrHashParams::default_fast();
        let alpha = if keys.len() < 100 {
            // Allocate twice as many slots as keys
            // Otherwise, the build could fail - see https://github.com/huggingface/tokenizers/pull/2675
            0.5
        } else {
            default.alpha
        };
        let params = PtrHashParams { alpha, ..default };
        let Some(hash) = FastPtrHash::try_new(keys, params) else {
            return Err(format!(
                "failed to build a perfect hash table over {} keys: ptr_hash found no placement",
                keys.len()
            )
            .into());
        };
        Ok(Self(hash))
    }

    #[inline(always)]
    pub(crate) fn index(&self, key: u64) -> usize {
        self.0.index(&key)
    }

    #[inline(always)]
    pub(crate) fn max_index(&self) -> usize {
        self.0.max_index()
    }
}

impl Default for Mphf {
    fn default() -> Self {
        // XXX: use Default implementation of PtrHash when fixed upstream
        Self::try_new(&[]).expect("always succeeds with empty key slice")
    }
}

/// Spreads an integer key over all 64 bits, for use as an [`Mphf`] key.
///
/// `mix` is a bijection, so distinct integers stay distinct keys.
#[inline(always)]
pub(crate) fn mix(z: u64) -> u64 {
    let z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^ (z >> 29)
}

/// The splitmix64 finalizer, a bijection like [`mix`]. On dense runs of integers, [`Mphf`] can fail
/// to build with [`mix`] keys (<https://github.com/huggingface/tokenizers/issues/2665>).
#[inline(always)]
pub(crate) fn splitmix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
