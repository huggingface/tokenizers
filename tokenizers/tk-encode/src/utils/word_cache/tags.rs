use wide::i8x16;

use super::cache::WordCache;
use crate::utils::signature::Signature;

#[inline(always)]
pub(super) fn tag_of(signature: Signature) -> u8 {
    // At least 0x01, otherwise it can be mistaken for an EMPTY slot
    ((signature.hash >> (64 - 8)) as u8).max(WordCache::EMPTY + 1)
}

pub(super) struct Window {
    window: [u8; WordCache::WINDOW_SIZE],
    offset: usize,
}

impl Window {
    pub(super) fn new(table: &[u8], start: usize) -> Self {
        Window {
            window: table[start..start + WordCache::WINDOW_SIZE]
                .try_into()
                .unwrap(),
            offset: start,
        }
    }

    /// Finds all candidates in the window that match the needle, and the leftmost 0x00 (empty slot)
    pub(super) fn find_matches_and_first_empty(&self, needle: u8) -> (SlotSet, Option<usize>) {
        let window = i8x16::from(self.window.map(|byte| byte as i8));
        let matches_bitmask = window.simd_eq(i8x16::from([needle as i8; 16])).to_bitmask() as u16;
        let empty_bitmask = window
            .simd_eq(i8x16::from([WordCache::EMPTY as i8; 16]))
            .to_bitmask() as u16;

        // a trick to truncate matches to before the first empty
        let before_first_empty = !empty_bitmask & empty_bitmask.wrapping_sub(1);

        let candidates = SlotSet {
            mask: matches_bitmask & before_first_empty,
            offset: self.offset,
        };
        let first_empty =
            (empty_bitmask != 0).then(|| self.offset + empty_bitmask.trailing_zeros() as usize);
        (candidates, first_empty)
    }
}

#[derive(Clone, Copy)]
pub(super) struct SlotSet {
    mask: u16,
    offset: usize,
}

impl Iterator for SlotSet {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.mask == 0 {
            return None;
        }
        let slot = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1;
        Some(slot + self.offset)
    }
}
