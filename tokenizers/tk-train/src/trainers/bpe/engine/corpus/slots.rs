//! Fixed-width slot planes and their joined read/write safety contract.
use super::super::WORD_SEPARATOR_ID;
use super::CorpusPlan;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, Ordering};
use tk_encode::Result;

// Reserve one code beyond admitted IDs for the separator. Layout is chosen
// once: sixteen bits for small vocabularies, three bytes through 24 bits, and
// the complete u32 domain beyond that.
pub(in super::super) fn slot_bits(id_count: usize) -> u8 {
    let required = (usize::BITS - id_count.leading_zeros()) as u8;
    if required <= 16 {
        16
    } else if required <= 24 {
        24
    } else {
        32
    }
}
// The coordinator chooses one storage type before training. Static dispatch
// keeps layout tests and variable-width bit arithmetic outside endpoint loops.
pub(in super::super) trait SlotStorage: Send + Sync + Sized {
    fn from_prepared(
        prepared: &CorpusPlan<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self>;
    fn len(&self) -> usize;
    fn load(&self, position: usize) -> u32;
    /// # Safety
    /// Stores belong to a joined write phase with no concurrent token readers.
    /// Each writer owns distinct logical slots until that phase joins.
    unsafe fn store(&self, position: usize, id: u32);
    #[cfg(target_arch = "x86_64")]
    fn prefetch_pointer(&self, position: usize) -> *const i8;
}
pub(in super::super) type U32Slots = Vec<AtomicU32>;
impl SlotStorage for U32Slots {
    fn from_prepared(
        prepared: &CorpusPlan<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        prepared.fill_tokens(workers, work, |_, id| AtomicU32::new(id))
    }
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        self[position].load(Ordering::Relaxed)
    }
    #[inline]
    unsafe fn store(&self, position: usize, id: u32) {
        self[position].store(id, Ordering::Relaxed);
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.as_ptr().wrapping_add(position).cast()
    }
}
pub(in super::super) struct U16Slots(Vec<AtomicU16>);
impl SlotStorage for U16Slots {
    fn from_prepared(
        prepared: &CorpusPlan<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        Ok(Self(prepared.fill_tokens(workers, work, |_, id| {
            AtomicU16::new(id as u16)
        })?))
    }
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        let id = self.0[position].load(Ordering::Relaxed);
        if id == u16::MAX {
            WORD_SEPARATOR_ID
        } else {
            u32::from(id)
        }
    }
    #[inline]
    unsafe fn store(&self, position: usize, id: u32) {
        debug_assert!(id == WORD_SEPARATOR_ID || id < u32::from(u16::MAX));
        self.0[position].store(id as u16, Ordering::Relaxed);
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.0.as_ptr().wrapping_add(position).cast()
    }
}
// The last initialized guard slot makes the final scalar four-byte read
// valid. It is outside the logical plane and is never rewritten.
pub(in super::super) struct PackedU24Slots {
    tokens: Vec<[AtomicU8; 3]>,
}
impl PackedU24Slots {
    const SEPARATOR: u32 = 0x00ff_ffff;
    fn bytes(id: u32) -> [AtomicU8; 3] {
        let bytes = id.to_le_bytes();
        std::array::from_fn(|index| AtomicU8::new(bytes[index]))
    }
}
impl SlotStorage for PackedU24Slots {
    fn from_prepared(
        prepared: &CorpusPlan<'_>,
        workers: usize,
        work: &crate::progress::WorkProgress,
    ) -> Result<Self> {
        let mut tokens = prepared.fill_tokens(workers, work, |_, id| Self::bytes(id))?;
        // fill_tokens reserves this guard before its parallel initialization,
        // avoiding a second allocation or a full-plane copy here.
        tokens.push(Self::bytes(0));
        Ok(Self { tokens })
    }
    #[inline]
    fn len(&self) -> usize {
        self.tokens.len() - 1
    }
    #[inline]
    fn load(&self, position: usize) -> u32 {
        assert!(position < self.len());
        // SAFETY: AtomicU8 has u8's size, alignment and valid representations.
        // All four bytes are initialized inside this allocation, including
        // the guard for the last slot. Writes require an exclusive joined
        // phase through unsafe store; concurrent read-only accesses are valid.
        // The next slot's low byte is masked off after this scalar read.
        let raw = unsafe {
            self.tokens
                .as_ptr()
                .add(position)
                .cast::<u32>()
                .read_unaligned()
        };
        let id = u32::from_le(raw) & Self::SEPARATOR;
        if id == Self::SEPARATOR {
            WORD_SEPARATOR_ID
        } else {
            id
        }
    }
    #[inline]
    unsafe fn store(&self, position: usize, id: u32) {
        debug_assert!(id == WORD_SEPARATOR_ID || id < Self::SEPARATOR);
        let bytes = id.to_le_bytes();
        let slot = &self.tokens[..self.len()][position];
        for index in 0..3 {
            slot[index].store(bytes[index], Ordering::Relaxed);
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn prefetch_pointer(&self, position: usize) -> *const i8 {
        self.tokens.as_ptr().wrapping_add(position).cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_slot_domains_preserve_ids_separators_and_adjacent_parallel_writes() {
        for (count, expected_bits) in [
            (0, 16),
            (65_535, 16),
            (65_536, 24),
            (16_777_215, 24),
            (16_777_216, 32),
            (u32::MAX as usize, 32),
            (usize::MAX, 32),
        ] {
            assert_eq!(slot_bits(count), expected_bits, "ID count {count}");
        }
        fn check<S: SlotStorage>(plane: S, count: usize) {
            const SLOTS: usize = 512;
            const ROUNDS: usize = 256;
            let mut values = vec![0, (count - 1) as u32, WORD_SEPARATOR_ID];
            values.extend(
                [65_534, 65_535, 65_536, 131_071, 131_072]
                    .into_iter()
                    .filter(|&id| (id as usize) < count),
            );
            std::thread::scope(|scope| {
                for lane in 0..8 {
                    let plane = &plane;
                    let values = &values;
                    scope.spawn(move || {
                        for round in 0..ROUNDS {
                            for position in (lane..SLOTS).step_by(8) {
                                // SAFETY: each thread owns its lane's slots;
                                // all reads follow this scope's join.
                                unsafe {
                                    plane
                                        .store(position, values[(position + round) % values.len()]);
                                }
                            }
                        }
                    });
                }
            });
            for position in 0..SLOTS {
                assert_eq!(
                    plane.load(position),
                    values[(position + ROUNDS - 1) % values.len()],
                    "count {count}, slot {position}"
                );
            }
        }
        check(
            PackedU24Slots {
                tokens: (0..513).map(|_| PackedU24Slots::bytes(0)).collect(),
            },
            16_777_215,
        );
        check(
            U16Slots((0..512).map(|_| AtomicU16::new(0)).collect()),
            65_535,
        );
        check(
            (0..512).map(|_| AtomicU32::new(0)).collect::<U32Slots>(),
            u32::MAX as usize,
        );
    }
}
