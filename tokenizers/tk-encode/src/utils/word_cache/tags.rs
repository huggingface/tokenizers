//! One byte of hash per slot, and the scan that finds the slots that may hold a word.

use wide::i8x16;

use crate::utils::signature::Signature;

/// The tag of a slot no word was stored in.
pub(super) const EMPTY: u8 = 0;

/// How many slots a word's window spans, starting at its home slot.
/// 16 tags are 16 bytes: a whole window fits in one vector register (SIMD)
/// and in one cache line, so scanning it for a hit can be a couple of instructions
/// and a single memory read.
pub(super) const WINDOW_SIZE: usize = 16;

#[inline(always)]
pub(super) fn tag_of(signature: Signature) -> u8 {
    // At least 0x01, otherwise it can be mistaken for an EMPTY slot
    ((signature.hash >> (64 - 8)) as u8).max(EMPTY + 1)
}

pub(super) struct Window {
    window: [u8; WINDOW_SIZE],
    offset: usize,
}

impl Window {
    /// The tags of the [`WINDOW_SIZE`] slots from `start`.
    pub(super) fn new(tags: &[u8], start: usize) -> Self {
        let window = tags[start..]
            .first_chunk()
            .expect("the table holds a whole window past its last home slot");
        Window {
            window: *window,
            offset: start,
        }
    }

    /// Finds all candidates in the window that match the needle, and the leftmost 0x00 (empty slot)
    pub(super) fn find_matches_and_first_empty(&self, needle: u8) -> (SlotSet, Option<usize>) {
        let window = i8x16::from(self.window.map(|byte| byte as i8));
        let matches_bitmask = window.simd_eq(i8x16::from([needle as i8; 16])).to_bitmask() as u16;
        let empty_bitmask = window.simd_eq(i8x16::from([EMPTY as i8; 16])).to_bitmask() as u16;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// One window shape per row: the needle in various lanes, an empty slot in
    /// the middle, at the edges, or absent. Candidates past the first empty
    /// lane must not be reported (no entry can live there, since inserts always
    /// fill the first empty lane) and the empty lane itself is the placement.
    #[test]
    fn the_scan_reports_matches_before_the_first_empty_and_the_empty_itself() {
        let offset = 3;
        let cases: &[(&[u8], u16, Option<usize>)] = &[
            // needle at lanes 0 and 2, empty at 3: lane 5's match is out of reach
            (
                &[
                    0xA7, 0x31, 0xA7, 0x00, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0x1A,
                ],
                0b101,
                Some(3),
            ),
            // no empty slot: every match is reachable, no placement
            (
                &[
                    0xA7, 0x31, 0xA7, 0x22, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0xA7,
                ],
                0b1000000000100101,
                None,
            ),
            // empty in lane 0: nothing is reachable
            (
                &[
                    0x00, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                    0xA7, 0xA7, 0xA7,
                ],
                0,
                Some(0),
            ),
            // every lane matches, empty in the last lane
            (
                &[
                    0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                    0xA7, 0xA7, 0x00,
                ],
                0b0111_1111_1111_1111,
                Some(15),
            ),
        ];
        for (lanes, want_mask, want_empty) in cases {
            let mut table = vec![0xEEu8; offset];
            table.extend_from_slice(lanes);
            let window = Window::new(&table, offset);
            let (candidates, first_empty) = window.find_matches_and_first_empty(0xA7);
            let found: Vec<usize> = candidates.collect();
            let want: Vec<usize> = (0..16)
                .filter(|i| want_mask >> i & 1 == 1)
                .map(|i| i + offset)
                .collect();
            assert_eq!(found, want, "lanes {lanes:?}");
            assert_eq!(
                first_empty,
                want_empty.map(|i| i + offset),
                "lanes {lanes:?}"
            );
        }
    }

    /// The tag row marks a free slot with 0x00, so no live entry may carry that
    /// tag: the scan stops looking at the first 0x00 it sees, and an insert
    /// treats the slot as free space. Enough words that a tag built from a raw
    /// hash byte would land on 0x00 hundreds of times.
    #[test]
    fn a_live_tag_is_never_the_empty_marker() {
        for i in 0..100_000u32 {
            let tag = tag_of(Signature::new(&i.to_le_bytes()));
            assert_ne!(tag, EMPTY, "word {i}");
        }
    }
}
