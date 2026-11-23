//! Where the ids and the bytes that do not fit in a slot are kept.

use super::ids::{CachedIds, MAX_RUN_IDS};
use crate::utils::uncovered::Uncovered;

/// Where a word's ids sit in a [`SpillBuffer`], and the generation they were stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Record {
    pub(super) start: u32,
    pub(super) end: u32,
    pub(super) generation: u32,
}

impl Record {
    #[inline(always)]
    pub(super) fn ids_len(self) -> usize {
        (self.end - self.start) as usize
    }
}

/// One buffer for every word a slot cannot hold whole: a word of more than three ids, or one
/// longer than its signature.
///
/// A record is a word's ids, then its [`Uncovered`] bytes when it has some, packed four to a
/// `u32`. The buffer ends with [`MAX_RUN_IDS`] zeros, so that many `u32`s can be read from the
/// start of any record.
///
/// Records are only added. When one does not fit in the budget, all of them are dropped at
/// once: the generation moves on, and a [`Record`] of an earlier generation is stale.
pub(super) struct SpillBuffer {
    buffer: Vec<u32>,
    /// The most `u32`s the records may take together.
    budget: usize,
    generation: u32,
}

impl SpillBuffer {
    pub(super) fn new(budget: usize) -> Self {
        Self {
            buffer: Vec::new(),
            // A record's offsets are `u32`s.
            budget: budget.min(u32::MAX as usize - MAX_RUN_IDS),
            generation: 0,
        }
    }

    /// The generation a [`Record`] has to be of for its ids to still be here.
    #[inline(always)]
    pub(super) fn generation(&self) -> u32 {
        self.generation
    }

    /// Stores a word's ids and its uncovered bytes.
    ///
    /// The `bool` is true when the generation wrapped back to zero. A record of any earlier
    /// generation would then pass for a current one, so the caller drops every record it holds.
    pub(super) fn store(
        &mut self,
        ids: impl ExactSizeIterator<Item = u32>,
        uncovered: Option<Uncovered<'_>>,
    ) -> (Record, bool) {
        let bytes = uncovered.map_or(&[][..], Uncovered::as_bytes);
        // The zeros at the end come off, and go back on after this record.
        let used = self.buffer.len().saturating_sub(MAX_RUN_IDS);
        self.buffer.truncate(used);
        let needed = ids.len() + bytes.len().div_ceil(4);
        let mut wrapped = false;
        if used + needed > self.budget {
            debug!(
                "WordCache: spill budget ({} ids) exhausted; dropping {used} spilled ids",
                self.budget
            );
            self.buffer.clear();
            self.generation = self.generation.wrapping_add(1);
            wrapped = self.generation == 0;
        }
        let start = self.buffer.len();
        self.buffer.extend(ids);
        let end = self.buffer.len();
        // The bytes go in as bytes, zero padded to whole `u32`s: one copy, where packing them
        // four at a time was a loop per word.
        self.buffer.resize(end + bytes.len().div_ceil(4), 0);
        let stored: &mut [u8] = bytemuck::cast_slice_mut(&mut self.buffer[end..]);
        stored[..bytes.len()].copy_from_slice(bytes);
        self.buffer.resize(self.buffer.len() + MAX_RUN_IDS, 0);
        let record = Record {
            start: start as u32,
            end: end as u32,
            generation: self.generation,
        };
        (record, wrapped)
    }

    /// The ids of a record of the current generation.
    #[inline(always)]
    pub(super) fn ids(&self, record: Record) -> CachedIds<'_> {
        CachedIds {
            padded: &self.buffer[record.start as usize..],
            len: record.ids_len(),
        }
    }

    /// The first [`MAX_RUN_IDS`] `u32`s at a record of the current generation: its ids, then
    /// what is stored after them.
    #[inline(always)]
    pub(super) fn padded_ids(&self, record: Record) -> &[u32; MAX_RUN_IDS] {
        self.buffer[record.start as usize..]
            .first_chunk()
            .expect("the buffer ends with MAX_RUN_IDS zeros")
    }

    /// Whether the bytes stored after the ids of `record` are `uncovered`.
    #[inline(always)]
    pub(super) fn holds(&self, record: Record, uncovered: Uncovered<'_>) -> bool {
        uncovered.is_prefix_of(bytemuck::cast_slice(&self.buffer[record.end as usize..]))
    }

    #[cfg(test)]
    pub(super) fn set_generation(&mut self, generation: u32) {
        self.generation = generation;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A buffer that kept every record would grow with the text. Eight ids a record against a
    /// budget of a hundred: it has to drop its records every twelve.
    #[test]
    fn the_buffer_never_outgrows_its_budget() {
        let mut spill = SpillBuffer::new(100);
        for i in 0..1000u32 {
            spill.store([i; 8].into_iter(), None);
            assert!(spill.buffer.len() <= 100 + MAX_RUN_IDS, "after {i} records");
        }
        assert_eq!(spill.generation(), 1000 / 12);
    }
}
