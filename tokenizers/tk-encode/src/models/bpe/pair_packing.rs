use std::ops::{BitAnd, Shr};

use crate::Result;

/// Number of bits used to encode the left and right mergees (the key) in the u64 [`PackedMerge`]
pub const NUM_KEY_BITS: u32 = 20;
/// Number of bits used to encode the rank in the u64 [`PackedMerge`]
pub const NUM_RANK_BITS: u32 = 64 - 2 * NUM_KEY_BITS;

const MIN_KEY_LEADING_ZEROS: u32 = 32 - NUM_KEY_BITS;
const MIN_RANK_LEADING_ZEROS: u32 = 32 - NUM_RANK_BITS;

/// On a u64, masks the bits where the merge key is encoded
pub const KEY_MASK: u64 = (1 << 40) - 1;
/// On a u64, masks the bits where the merge rank is encoded
pub const RANK_MASK: u64 = u64::MAX ^ KEY_MASK;

/// Information about a BPE merge, encoded on 64 bits
/// - Fist 24 bits encode the rank of the merge
/// - The next 20 bits encode the ID of the left token
/// - The remaining 20 bits encode the ID of the right token
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackedMerge(u64);

impl PackedMerge {
    pub fn new(value: u64) -> Self {
        Self(value)
    }
}

impl From<u64> for PackedMerge {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl BitAnd<u64> for PackedMerge {
    type Output = Self;
    fn bitand(self, rhs: u64) -> Self::Output {
        Self(self.0 & rhs)
    }
}

impl Shr<u64> for PackedMerge {
    type Output = u64;

    fn shr(self, rhs: u64) -> u64 {
        self.0 >> rhs
    }
}

/// Packs both u32 (token id) values in a single u64 key
/// Assumes the vocab has cardinality < 2^NUM_KEY_BITS (ie 2^20 = 1 048 576)
///
///
/// Example:
/// ```
/// use std::assert_matches;
/// use tk_encode::models::bpe::pair_packing::{PackedMerge, make_packed_merge};
///
/// let left_token: u32 = 0x00_0F_2A_12;
/// let right_token: u32 = 0x00_0C_22_E1;
/// let packed = make_packed_merge(left_token, right_token, 0x00);
/// let expected = PackedMerge::new(0x000000_F2A12_C22E1);
/// //                               |      |^^^^^|     | left_token (20 bits)
/// //.                              |      |     |^^^^^| right_token (20 bits)
/// //                               |^^^^^^|     |     | 24 bits headway to encode something else (rank)
/// assert_matches!(
/// 	packed,
/// 	Ok(value) if value == expected
/// );
/// ```
///
#[inline(always)]
pub fn make_packed_merge(
    left_token: u32,
    right_token: u32,
    merge_rank: u32,
) -> Result<PackedMerge> {
    if left_token >= (1 << NUM_KEY_BITS) {
        return Err(format!(
            "left_token must have at least {} leading zeros",
            &MIN_KEY_LEADING_ZEROS
        )
        .into());
    }
    if right_token >= (1 << NUM_KEY_BITS) {
        return Err(format!(
            "right_token must have at least {} leading zeros",
            &MIN_KEY_LEADING_ZEROS
        )
        .into());
    }
    if merge_rank >= (1 << NUM_RANK_BITS) {
        return Err(format!(
            "merge_rank must have at least {} leading zeros",
            &MIN_KEY_LEADING_ZEROS
        )
        .into());
    }
    Ok(PackedMerge(
        ((left_token as u64) << NUM_KEY_BITS) | right_token as u64,
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use std::assert_matches;

    #[test]
    fn test_packing() {
        for (a, b, expected) in [
            (0x1BCE3, 0x22FA3, 0x1BCE322FA3),
            (0x22FA3, 0x1BCE3, 0x22FA31BCE3),
            (0xFFFFF, 0x0, 0xFFFFF00000),
            (0x0, 0xFFFFF, 0x00000FFFFF),
            (0x0, 0x0, 0x0),
        ] {
            let packed = make_packed_merge(a, b, 0x0);
            assert_matches!(
                packed,
                Ok(value) if value == PackedMerge(expected),
                "Packing {a:#04X} and {b:#04X} is expected to produce {expected:#04X}, but got {packed:?}"
            );
        }
    }

    #[test]
    fn test_invariant_by_mask() {
        for (a, b) in [
            (0x1BCE3, 0x22FA3),
            (0x22FA3, 0x1BCE3),
            (0xFFFFF, 0x0),
            (0x0, 0xFFFFF),
            (0x0, 0x0),
            (0xFFFFF, 0xFFFFF),
        ] {
            let packed = make_packed_merge(a, b, 0x0).unwrap();
            assert_eq!(packed, packed & KEY_MASK);
        }
    }

    #[test]
    fn test_invalid_arguments() {
        // left and right values are too large, should fail
        for (a, b) in [(0xFFFFFF, 0x12AE), (0x12AE, 0xFFFFFF)] {
            let packed = make_packed_merge(a, b, 0x0);
            assert_matches!(packed, Err(_),);
        }

        assert_matches!(make_packed_merge(0x0, 0x0, u32::MAX), Err(_))
    }
}
