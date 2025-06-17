/// Small helper to manage a bit field which supports predecessor and successor queries with a simple scan implementation.
/// This is sufficient for our use case, since two one bits will be at most 128 bits apart.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BitField {
    bitfield: Vec<u64>,
}

impl BitField {
    /// All bits are initialized to 1.
    pub(crate) fn new(bits: usize) -> Self {
        Self {
            bitfield: vec![u64::MAX; (bits + 63) / 64],
        }
    }

    pub(crate) fn is_set(&self, bit: usize) -> bool {
        let (word, bit) = (bit / 64, bit % 64);
        self.bitfield[word] & (1 << bit) != 0
    }

    pub(crate) fn clear(&mut self, bit: usize) {
        let (word, bit) = (bit / 64, bit % 64);
        self.bitfield[word] &= !(1 << bit);
    }

    pub(crate) fn successor(&self, bit: usize) -> usize {
        let (mut word_idx, bit_idx) = (bit / 64, bit % 64);
        let word = self.bitfield[word_idx] >> bit_idx;
        if word != 0 {
            word.trailing_zeros() as usize + bit
        } else {
            loop {
                word_idx += 1;
                let word = self.bitfield[word_idx];
                if word != 0 {
                    break word.trailing_zeros() as usize + word_idx * 64;
                }
            }
        }
    }

    pub(crate) fn predecessor(&self, bit: usize) -> usize {
        let (mut word_idx, bit_idx) = (bit / 64, bit % 64);
        let word = self.bitfield[word_idx] << (63 - bit_idx);
        if word != 0 {
            bit - word.leading_zeros() as usize
        } else {
            loop {
                word_idx -= 1;
                let word = self.bitfield[word_idx];
                if word != 0 {
                    break word_idx * 64 + 63 - word.leading_zeros() as usize;
                }
            }
        }
    }
}
