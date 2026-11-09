//! The perfect hash behind the vocabulary, the BPE merge table and the other fixed lookup tables.
//!
//! A perfect hash is built once from a fixed set of keys and sends each of them to its own slot, so
//! there are no collisions to resolve. A key outside the set still lands on some slot, so a table
//! keeps the key (or something derived from it) in the slot and compares it on lookup.
//!
//! [`Mphf`] takes keys that are already hashed. Integer keys go through [`mix`] first, or through
//! [`mix_dense`] when they are a dense run, such as token ids.

use ptr_hash::{FastPtrHash, hash::NoHash};

/// The one perfect-hash type in the crate. `FastPtrHash` is generic, so each other instantiation
/// would add another copy of its code to the binary.
pub(crate) type Mphf = FastPtrHash<NoHash, u64>;

/// Spreads an integer key over all 64 bits, for use as an [`Mphf`] key.
///
/// `mix` is a bijection, so distinct integers stay distinct keys.
#[inline(always)]
pub(crate) fn mix(z: u64) -> u64 {
    let z = z.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^ (z >> 29)
}

/// [`mix`] for a dense run of integers, such as the ids of the added tokens.
///
/// One multiply does not spread consecutive integers enough for [`Mphf`]: on the ids `0..757`, its
/// construction fails, and panics, on almost every attempt. This is the splitmix64 finalizer, also
/// a bijection. It costs two more multiplies, so the per-word vocabulary lookup keeps [`mix`].
#[inline(always)]
pub(crate) fn mix_dense(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
