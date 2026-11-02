//! The perfect hash behind the vocabulary, the BPE merge table and the other fixed lookup tables.
//!
//! A perfect hash is built once from a fixed set of keys and sends each of them to its own slot, so
//! there are no collisions to resolve. A key outside the set still lands on some slot, so a table
//! keeps the key (or something derived from it) in the slot and compares it on lookup.
//!
//! [`Mphf`] takes keys that are already hashed. Integer keys go through [`mix`] first.

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
