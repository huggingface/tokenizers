//! A table that remembers which token ids a word encodes to, so the tokenization
//! model (the expensive step) only encodes each word once.
//!
//! A word is looked up by its [`Signature`]: its length, first eight bytes and last eight bytes.
//! A hash of the signature places the word in the table:
//! - the bottom bits pick its **home slot**. A word can be cached in the 16-slot window that starts there
//! - the top byte is a **tag**, stored in a separate table ([`WordCache::quick_lookup`])
//!
//! A lookup reads the home slot first: most cached words sit there, and a vacant one ends it.
//! Otherwise it walks a 16 byte window in [`WordCache::quick_lookup`] to find a matching tag.
//! When a tag matches, the slot's signature confirms whether it's a hit or not.
//!
//! On miss, we return where the cache should insert ([`WordCache::insert`]) the ids;
//! Either the slot already holding a stale copy of the word ([`WordCacheSlot::is_stale`]),
//! the first empty (0x00) slot in the window, or the home slot if the window is full.
//!
//! ```text
//!    lookup "hat":  tag A7, home slot 5
//!
//!    slot:         4     5     6     7
//!               ┌─────┬─────┬─────┬─────┬────
//!    tags       │ C4  │ A7  │ 31  │ A7  │ ...     one hash byte per slot
//!               └─────┴─────┴─────┴─────┴────
//!               ┌─────┬─────┬─────┬─────┬────
//!    slots      │"cat"│"the"│"sat"│"hat"│ ...     the signature and the ids, 32 bytes
//!               └─────┴─────┴─────┴─────┴────
//!                        ▲           ▲
//!                        │           └ tag and signature match: a hit, return the ids
//!                        └ same tag, wrong signature: keep walking
//! ```
//!
//! A slot ([`WordCacheSlot`]) keeps up to three ids inline.
//! Longer encodings go to one shared buffer ([`WordCache::spilled_buffer`]) and the slot holds offsets in that buffer.
//!
//! # A hit is always the word that was stored
//!
//! A signature holds a word of [`Signature::BYTES`] bytes or less whole, so a matching slot is the same word.
//! A longer word also stores its [`Uncovered`] bytes, in [`WordCache::spilled_buffer`] after its ids.
//! A lookup compares them before it reports a hit.
//!
//! # Where the ideas come from
//!
//! - [Swiss Tables] is where the tag row comes from: one byte of hash per slot,
//!   checked before the slot itself is touched.
//! - [gigatoken] is a BPE tokenizer with a pre-token cache built from the same
//!   parts: packed keys, 32-byte self-contained slots, ids inline.
//! - [huggingface/tokenizers#2234] is an open-addressed cache for this same encode
//!   pipeline, arrived at in parallel.
//!
//! [Swiss Tables]: https://abseil.io/about/design/swisstables
//! [gigatoken]: https://github.com/marcelroed/gigatoken
//! [huggingface/tokenizers#2234]: https://github.com/huggingface/tokenizers/pull/2234

mod cache;
mod ids;
mod run;
mod slot;
mod tags;
#[cfg(test)]
mod tests;

pub use cache::{Lookup, WordCache};
pub use ids::MAX_RUN_IDS;

#[cfg(doc)]
use crate::utils::signature::Signature;
#[cfg(doc)]
use slot::WordCacheSlot;
