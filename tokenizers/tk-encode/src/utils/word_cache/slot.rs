//! One slot of the word cache.

use super::ids::{CachedIds, MAX_RUN_IDS};
use super::spill::Record;
#[cfg(doc)]
use super::spill::SpillBuffer;
#[cfg(doc)]
use super::table::WordCache;
use crate::utils::signature::Signature;
#[cfg(doc)]
use crate::utils::uncovered::Uncovered;

pub(super) const MAX_INLINE_IDS: usize = 3;

/// A cache slot, packed in 32 bytes.
/// There are two variants:
///
/// ## inline: 3 or fewer ids to cache, for a word of [`Signature::BYTES`] bytes or less
///
/// The token ids are encoded inline in the cache slot:
/// ```text
/// ┌───────────┬───────────┬──────────┬──────────┬──────────┬──────────────────────────┐
/// │   head    │   tail    │  id[0]   │  id[1]   │  id[2]   │           meta           │
/// │    u64    │    u64    │   u32    │   u32    │   u32    │ word_len << 8 | ids_len  │
/// └───────────┴───────────┴──────────┴──────────┴──────────┴──────────────────────────┘
/// 0           8           16         20         24         28                       32 bytes
/// ```
///
/// ## spilled: every other word
///
/// The token ids are stored in a [`SpillBuffer`], followed by the word's [`Uncovered`] bytes.
/// The slot holds the [`Record`] that says where.
/// ```text
/// ┌───────────┬───────────┬──────────┬──────────┬────────────┬──────────────────────────┐
/// │   head    │   tail    │  start   │   end    │ generation │           meta           │
/// │    u64    │    u64    │   u32    │   u32    │    u32     │ word_len << 8 | SPILLED  │
/// └───────────┴───────────┴──────────┴──────────┴────────────┴──────────────────────────┘
/// 0           8           16         20         24           28                       32 bytes
/// ```
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct WordCacheSlot {
    head: u64,
    tail: u64,
    payload: [u32; 3],
    /// The word's length in the top 24 bits. The number of inline ids in the low 8, or
    /// [`Self::SPILLED`].
    meta: u32,
}

const _: () = assert!(
    (MAX_INLINE_IDS + 1).is_power_of_two(),
    "WordCacheSlot::INLINE_IDS is a bit mask: it needs MAX_INLINE_IDS to be all ones"
);
// 32-byte size and alignment, so a read is always contained in a cache line
const _: () = assert!(size_of::<WordCacheSlot>() == 32);
const _: () = assert!(align_of::<WordCacheSlot>() == 32);

impl WordCacheSlot {
    /// What the low byte of `meta` holds when the ids are in the spill buffer.
    const SPILLED: u32 = 0xFF;

    /// The longest word whose length fits in `meta`.
    pub(super) const MAX_WORD_BYTES: usize = (1 << 24) - 1;

    /// The bits of `meta` that hold up to [`MAX_INLINE_IDS`]: an inline hit ignores them.
    const INLINE_IDS: u64 = MAX_INLINE_IDS as u64;

    fn new(signature: Signature, payload: [u32; 3], ids_len: u32) -> Self {
        Self {
            head: signature.head,
            tail: signature.tail,
            payload,
            meta: (signature.len as u32) << 8 | ids_len,
        }
    }

    /// A slot that holds the ids itself: [`MAX_INLINE_IDS`] at most.
    pub(super) fn inline(signature: Signature, ids: impl ExactSizeIterator<Item = u32>) -> Self {
        debug_assert!(ids.len() <= MAX_INLINE_IDS);
        let ids_len = ids.len() as u32;
        let mut payload = [0u32; 3];
        for (lane, id) in payload.iter_mut().zip(ids) {
            *lane = id;
        }
        Self::new(signature, payload, ids_len)
    }

    /// A slot whose ids are in the spill buffer, at `record`.
    pub(super) fn spilled(signature: Signature, record: Record) -> Self {
        let payload = [record.start, record.end, record.generation];
        Self::new(signature, payload, Self::SPILLED)
    }

    /// Whether no insert wrote this slot since the table was created or cleared: it is all zeros,
    /// which [`WordCache::insert`] never writes.
    #[inline(always)]
    pub(super) fn is_vacant(&self) -> bool {
        self.head == 0 && self.tail == 0 && self.meta == 0
    }

    // This and the two below compare lengths as `u64`: a word too long for `meta` then matches
    // no slot.
    #[inline(always)]
    pub(super) fn has_signature(&self, signature: Signature) -> bool {
        self.head == signature.head
            && self.tail == signature.tail
            && u64::from(self.meta >> 8) == signature.len as u64
    }

    /// [`Self::has_signature`] with the ids inline, in one compare of `meta`.
    /// Such a slot never holds a word longer than its signature: [`WordCache::insert`] spills it.
    #[inline(always)]
    pub(super) fn holds_inline(&self, signature: Signature) -> bool {
        self.head == signature.head
            && self.tail == signature.tail
            && u64::from(self.meta) & !Self::INLINE_IDS == (signature.len as u64) << 8
    }

    /// [`Self::has_signature`] with a record [`WordCache::emit_home_hits`] may answer with: of
    /// `generation`, and of no more than [`MAX_RUN_IDS`] ids.
    #[inline(always)]
    pub(super) fn run_record(&self, signature: Signature, generation: u32) -> Option<Record> {
        let [start, end, stored_in] = self.payload;
        (self.head == signature.head
            && self.tail == signature.tail
            && u64::from(self.meta) == (signature.len as u64) << 8 | u64::from(Self::SPILLED)
            && stored_in == generation
            && (end - start) as usize <= MAX_RUN_IDS)
            .then_some(Record {
                start,
                end,
                generation,
            })
    }

    /// Where the ids are in the spill buffer. `None` when the slot holds them itself.
    #[inline(always)]
    pub(super) fn record(&self) -> Option<Record> {
        let [start, end, generation] = self.payload;
        (self.meta & 0xFF == Self::SPILLED).then_some(Record {
            start,
            end,
            generation,
        })
    }

    /// The ids of a slot that holds them itself.
    #[inline(always)]
    pub(super) fn inline_ids(&self) -> CachedIds<'_> {
        CachedIds {
            padded: &self.payload,
            len: (self.meta & 0xFF) as usize,
        }
    }

    /// Copies the three id lanes of a slot that holds its ids to `dst`, whatever their number.
    /// Returns `dst` moved past the ids the slot holds.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of [`MAX_INLINE_IDS`] `u32`s.
    #[inline(always)]
    pub(super) unsafe fn copy_inline_ids(&self, dst: *mut u32) -> *mut u32 {
        // SAFETY: the caller left room for the three lanes, and the count is three at most.
        unsafe {
            dst.copy_from_nonoverlapping(self.payload.as_ptr(), MAX_INLINE_IDS);
            dst.add((self.meta & 0xFF) as usize)
        }
    }
}
