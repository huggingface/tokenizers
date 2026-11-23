#[cfg(doc)]
use super::cache::WordCache;
use super::ids::{CachedIds, MAX_RUN_IDS};
use crate::utils::signature::Signature;

pub const MAX_INLINE_IDS: usize = 3;

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
/// The token ids are stored in [`WordCache::spilled_buffer`], followed by the word's
/// [`Uncovered`] bytes. The slot holds the offsets of the ids in that buffer.
/// ```text
/// ┌───────────┬───────────┬──────────┬──────────┬────────────┬──────────────────────────┐
/// │   head    │   tail    │  start   │   end    │ generation │           meta           │
/// │    u64    │    u64    │   u32    │   u32    │    u32     │ word_len << 8 | SPILLED  │
/// └───────────┴───────────┴──────────┴──────────┴────────────┴──────────────────────────┘
/// 0           8           16         20         24           28                       32 bytes
/// ```
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WordCacheSlot {
    pub(super) head: u64,
    pub(super) tail: u64,
    pub(super) payload: [u32; 3],
    /// The word's length in the top 24 bits. The number of inline ids in the low 8, or
    /// [`Self::SPILLED`].
    pub(super) meta: u32,
}

const _: () = assert!(
    (MAX_INLINE_IDS + 1).is_power_of_two(),
    "WordCacheSlot::INLINE_IDS is a bit mask: it needs MAX_INLINE_IDS to be all ones"
);
// 32-byte size and alignment, so a read is always contained in a cache line
const _: () = assert!(size_of::<WordCacheSlot>() == 32);
const _: () = assert!(align_of::<WordCacheSlot>() == 32);

impl WordCacheSlot {
    /// A sentinel value that discriminates an inline slot (ids are stored in the slot) from a spilled slot
    /// (values are stored in a buffer)
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

    pub fn new_spilled(signature: Signature, ids_offsets: (usize, usize), generation: u32) -> Self {
        let (start, end) = ids_offsets;
        Self::new(
            signature,
            [start as u32, end as u32, generation],
            Self::SPILLED,
        )
    }

    pub fn new_self_contained(
        signature: Signature,
        ids: impl ExactSizeIterator<Item = u32>,
    ) -> Self {
        assert!(ids.len() <= MAX_INLINE_IDS);
        let ids_len = ids.len() as u32;
        let mut payload = [0u32; 3];
        for (slot, id) in payload.iter_mut().zip(ids) {
            *slot = id;
        }
        Self::new(signature, payload, ids_len)
    }

    pub fn is_stale(&self, current_generation: u32) -> bool {
        match self.variant() {
            CacheSlotType::SelfContained(_) => false,
            CacheSlotType::Spilled(spilled) => spilled.generation() != current_generation,
        }
    }
}

impl WordCacheSlot {
    // Both compare lengths as `u64`: a word too long for `meta` then matches no slot.
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

    /// [`Self::has_signature`] with ids in the buffer that [`WordCache::emit_home_hits`] may
    /// answer with: stored at `generation`, and no more than [`MAX_RUN_IDS`] of them.
    #[inline(always)]
    pub(super) fn holds_spilled(&self, signature: Signature, generation: u32) -> bool {
        let [start, end, stored_at] = self.payload;
        self.head == signature.head
            && self.tail == signature.tail
            && u64::from(self.meta) == (signature.len as u64) << 8 | u64::from(Self::SPILLED)
            && stored_at == generation
            && (end - start) as usize <= MAX_RUN_IDS
    }

    /// Whether no insert wrote this slot since the table was created or cleared: it is all zeros,
    /// which [`WordCache::insert`] never writes.
    #[inline(always)]
    pub(super) fn is_vacant(&self) -> bool {
        self.head == 0 && self.tail == 0 && self.meta == 0
    }

    #[inline(always)]
    pub(super) fn ids_len(&self) -> usize {
        (self.meta & 0xFF) as usize
    }

    fn is_spilled(self) -> bool {
        self.meta & 0xFF == Self::SPILLED
    }

    pub(super) fn variant<'a>(&'a self) -> CacheSlotType<'a> {
        if self.is_spilled() {
            CacheSlotType::Spilled(Spilled(self))
        } else {
            CacheSlotType::SelfContained(SelfContained(self))
        }
    }
}

/// Convenience wrapper around [`WordCacheSlot`] to discriminate
/// whether it's a spilled or self-contained slot
pub enum CacheSlotType<'a> {
    SelfContained(SelfContained<'a>),
    Spilled(Spilled<'a>),
}

#[repr(transparent)]
pub struct Spilled<'a>(pub(super) &'a WordCacheSlot);

impl<'a> Spilled<'a> {
    pub(super) fn ids_offsets(self) -> (usize, usize) {
        let [start, end, _] = self.0.payload;
        (start as usize, end as usize)
    }

    pub(super) fn generation(self) -> u32 {
        self.0.payload[2]
    }
}

#[repr(C, align(32))]
pub struct SelfContained<'a>(&'a WordCacheSlot);

impl<'a> SelfContained<'a> {
    pub(super) fn ids(self) -> CachedIds<'a> {
        CachedIds {
            padded: &self.0.payload,
            len: self.0.ids_len(),
        }
    }
}
