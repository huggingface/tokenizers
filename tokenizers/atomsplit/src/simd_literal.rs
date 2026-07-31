//! Finds every place a short pattern (up to three bytes) occurs in a text, in one pass.
//! This is the scan behind [`Literal::matches_into`](crate::literal::Literal::matches_into);
//! [`crate::literal`] explains when it is picked over the plain iterator.
//!
//! # Why a dedicated scan
//!
//! A pre-tokenizer splits on a delimiter that shows up every few bytes: running English text
//! has a space about every six bytes, SentencePiece text has a `▁` at every word. A
//! next-match engine like `memmem` is built for the opposite case, a needle that is rare, so
//! at this density it spends most of its time stopping and restarting: every match returns to
//! the caller, and every call re-enters the search machinery. This scan never stops. It
//! answers "where are ALL the matches in these 16 bytes?" with a handful of instructions,
//! block after block.
//!
//! # How: compare, mask, decode
//!
//! Two words come up in everything below. A SIMD **register** is an extra-wide CPU value
//! holding 16 bytes side by side; one instruction operates on all 16 at once. Each of those
//! 16 byte slots is a **lane**. A SIMD compare answers lane by lane: a lane becomes all ones
//! where the answer is yes, all zeros where it is no.
//!
//! **Step 1, compare.** Compare the text against the pattern's first byte (copied across all
//! 16 lanes), the text shifted by one against the second byte, and so on, then AND the
//! answers together: a lane that passed every compare is the start of a whole match, and
//! nothing needs a second look. Searching for `▁` (three bytes, `E2 96 81`) in
//! `"a…b▁c"`, where the `…` (`E2 80 A6`) shares a first byte with `▁` and acts as the decoy:
//!
//! ```text
//!   offset               0    1    2    3    4    5    6    7    8
//!   text                 a    E2   80   A6   b    E2   96   81   c
//!                             └── … ──────┘       └── ▁ ──────┘
//!   text[j]   == E2 ?    ·    ✓    ·    ·    ·    ✓    ·    ·    ·
//!   text[j+1] == 96 ?    ·    ·    ·    ·    ·    ✓    ·    ·    ·
//!   text[j+2] == 81 ?    ·    ·    ·    ·    ·    ✓    ·    ·    ·
//!   AND of the three     ·    ·    ·    ·    ·    ✓    ·    ·    ·
//! ```
//!
//! Only offset 5 passes all three rows: a match starts at 5, and the decoy at 1 is out after
//! the second row.
//!
//! **Step 2, mask.** The 16 lane answers are squeezed into one ordinary integer so plain
//! arithmetic can take over: bit `j` set means "a match starts at offset `j`". For the
//! example above the mask is `0b100000`, bit 5. Each target has a one-or-two instruction way
//! to build this integer; on NEON the bits come out spaced four positions apart instead of
//! one, a spacing the shared code carries along as `SHIFT` (the per-target layer below).
//!
//! **Step 3, decode.** Count-trailing-zeros on the mask gives the lowest set bit, which is the
//! next match offset; clear that bit and repeat. The one twist is that the code decodes four
//! offsets *without checking whether any bits are left*. Say a block starting at text offset
//! `base` had matches at offsets 2 and 5 (shown with the one-bit spacing):
//!
//! ```text
//!   mask 0b100100  trailing zeros = 2   write base + 2   clear bit → 0b100000
//!   mask 0b100000  trailing zeros = 5   write base + 5   clear bit → 0
//!   mask 0       trailing zeros = 64  write garbage    (slot 3)
//!   mask 0       trailing zeros = 64  write garbage    (slot 4)
//!                                      cursor += 2      (how many bits were set)
//! ```
//!
//! The two garbage slots sit past the cursor, so the next block's writes (or nothing) land on
//! them and they never reach the caller. The honest alternative, checking "still something
//! left?" before each decode, puts a branch in the hottest loop; with a match every four to
//! eight bytes that branch guesses wrong about once per block, which costs more than all the
//! compares together.
//!
//! Sparse text takes none of this: one instruction per 64-byte block answers "nothing here"
//! (see `any`), so a pattern that is rare or absent stays close to `memmem` speed.
//!
//! The compare trick is Wojciech Muła's ("SIMD-friendly algorithms for substring searching");
//! the branch-free decode is how simdjson reads its masks.
//!
//! # The per-target layer
//!
//! Everything from the mask on is plain integer arithmetic and is shared. A target provides
//! four primitives in its `arch` module:
//!
//! - `splat`: one byte copied into every lane, the shape a compare wants.
//! - `match_starts`: step 1, the shifted compares ANDed together.
//! - `any`: "did anything in these 64 bytes match?", the cheap exit for sparse text.
//! - `bits`: step 2, lane answers to integer mask.
//!
//! x86 (`movemask`) and wasm (`bitmask`) each have an instruction that grabs one bit from
//! every lane, so offset `j` lands on bit `j`. NEON has no such instruction; its standard
//! substitute (Danila Kutenin's) leaves the bits four positions apart, so offset `j` lands
//! on bit `4 * j`. `SHIFT` is that spacing (`bit position >> SHIFT` recovers the offset),
//! and it is the only difference the shared code ever sees.
//!
//! The wasm layer is compile-checked but not run in CI; the other targets run the full
//! `tests/literal.rs` parity suite.

#[cfg(target_arch = "aarch64")]
mod arch {
    use core::arch::aarch64::*;

    /// 16 bytes of text in one register.
    pub(super) type Chunk = uint8x16_t;
    /// Offsets sit four bit positions apart in a mask from [`bits`]: offset `j` is bit `4 * j`.
    pub(super) const SHIFT: u32 = 2;

    /// One byte copied into every lane.
    #[inline(always)]
    pub(super) fn splat(byte: u8) -> Chunk {
        unsafe { vdupq_n_u8(byte) }
    }

    /// All-ones in every lane where a whole `K`-byte match starts.
    ///
    /// # Safety
    /// `p .. p + 16 + K - 1` must be readable: the shifted compares load 16 bytes from each
    /// of `p`, `p + 1`, .., `p + K - 1`.
    #[inline(always)]
    pub(super) unsafe fn match_starts<const K: usize>(p: *const u8, pattern: &[Chunk; K]) -> Chunk {
        // SAFETY: the caller's contract above; each load of 16 bytes from `p + k`, `k < K`,
        // stays inside that readable range.
        let mut starts = unsafe { vceqq_u8(vld1q_u8(p), pattern[0]) };
        for (k, &byte) in pattern.iter().enumerate().skip(1) {
            starts = unsafe { vandq_u8(starts, vceqq_u8(vld1q_u8(p.add(k)), byte)) };
        }
        starts
    }

    /// `true` when anything matched in the four chunks.
    #[inline(always)]
    pub(super) fn any(s0: Chunk, s1: Chunk, s2: Chunk, s3: Chunk) -> bool {
        unsafe { vmaxvq_u8(vorrq_u8(vorrq_u8(s0, s1), vorrq_u8(s2, s3))) != 0 }
    }

    /// The lane answers as one integer: one bit per lane, four bit positions apart.
    ///
    /// NEON cannot grab one bit from every lane in a single instruction. Instead `vshrn`
    /// ("shift right and narrow") halves the register to 64 bits, shrinking each lane's
    /// answer from eight bits to four (still all ones or all zeros); the `0x1111..` mask
    /// then keeps one bit of each four, so clearing one bit in the decode drops exactly
    /// one match.
    #[inline(always)]
    pub(super) fn bits(starts: Chunk) -> u64 {
        unsafe {
            let halved = vshrn_n_u16::<4>(vreinterpretq_u16_u8(starts));
            vget_lane_u64::<0>(vreinterpret_u64_u8(halved)) & 0x1111_1111_1111_1111
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use core::arch::x86_64::*;

    /// 16 bytes of text in one register.
    pub(super) type Chunk = __m128i;
    /// One bit per lane in a mask from [`bits`]: offset `j` is bit `j`, nothing to shift.
    pub(super) const SHIFT: u32 = 0;

    /// One byte copied into every lane.
    #[inline(always)]
    pub(super) fn splat(byte: u8) -> Chunk {
        unsafe { _mm_set1_epi8(byte as i8) }
    }

    /// All-ones in every lane where a whole `K`-byte match starts.
    ///
    /// # Safety
    /// `p .. p + 16 + K - 1` must be readable: the shifted compares load 16 bytes from each
    /// of `p`, `p + 1`, .., `p + K - 1`.
    #[inline(always)]
    pub(super) unsafe fn match_starts<const K: usize>(p: *const u8, pattern: &[Chunk; K]) -> Chunk {
        // SAFETY: the caller's contract above; each load of 16 bytes from `p + k`, `k < K`,
        // stays inside that readable range.
        let mut starts = unsafe { _mm_cmpeq_epi8(_mm_loadu_si128(p.cast()), pattern[0]) };
        for (k, &byte) in pattern.iter().enumerate().skip(1) {
            starts = unsafe {
                _mm_and_si128(
                    starts,
                    _mm_cmpeq_epi8(_mm_loadu_si128(p.add(k).cast()), byte),
                )
            };
        }
        starts
    }

    /// `true` when anything matched in the four chunks.
    #[inline(always)]
    pub(super) fn any(s0: Chunk, s1: Chunk, s2: Chunk, s3: Chunk) -> bool {
        unsafe { _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(s0, s1), _mm_or_si128(s2, s3))) != 0 }
    }

    /// The lane answers as one integer: `movemask` grabs each lane's top bit, so offset `j`
    /// is bit `j`.
    #[inline(always)]
    pub(super) fn bits(starts: Chunk) -> u64 {
        unsafe { _mm_movemask_epi8(starts) as u32 as u64 }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod arch {
    use core::arch::wasm32::*;

    /// 16 bytes of text in one register.
    pub(super) type Chunk = v128;
    /// One bit per lane in a mask from [`bits`]: offset `j` is bit `j`, nothing to shift.
    pub(super) const SHIFT: u32 = 0;

    /// One byte copied into every lane.
    #[inline(always)]
    pub(super) fn splat(byte: u8) -> Chunk {
        u8x16_splat(byte)
    }

    /// All-ones in every lane where a whole `K`-byte match starts.
    ///
    /// # Safety
    /// `p .. p + 16 + K - 1` must be readable: the shifted compares load 16 bytes from each
    /// of `p`, `p + 1`, .., `p + K - 1`.
    #[inline(always)]
    pub(super) unsafe fn match_starts<const K: usize>(p: *const u8, pattern: &[Chunk; K]) -> Chunk {
        // SAFETY: the caller's contract above; each load of 16 bytes from `p + k`, `k < K`,
        // stays inside that readable range.
        let mut starts = unsafe { u8x16_eq(v128_load(p.cast()), pattern[0]) };
        for (k, &byte) in pattern.iter().enumerate().skip(1) {
            starts = unsafe { v128_and(starts, u8x16_eq(v128_load(p.add(k).cast()), byte)) };
        }
        starts
    }

    /// `true` when anything matched in the four chunks.
    #[inline(always)]
    pub(super) fn any(s0: Chunk, s1: Chunk, s2: Chunk, s3: Chunk) -> bool {
        v128_any_true(v128_or(v128_or(s0, s1), v128_or(s2, s3)))
    }

    /// The lane answers as one integer: `bitmask` grabs each lane's top bit, so offset `j`
    /// is bit `j`.
    #[inline(always)]
    pub(super) fn bits(starts: Chunk) -> u64 {
        u8x16_bitmask(starts) as u64
    }
}

/// Decodes every set bit of `bits` into a `base + offset` value at `cursor`, and returns the
/// cursor advanced by the true match count.
///
/// The first four decodes run without checking `bits` (the module docs say why): once the mask
/// is used up, `trailing_zeros` is 64 and the slot gets a garbage offset, scratch that the
/// count never covers and a later call overwrites. Blocks with more than four matches finish
/// in the loop.
///
/// # Safety
/// There must be room for `max(4, count_ones(bits))` writes between `cursor` and `end`: four
/// slots for the unconditional decodes, one per match when a block has more. Debug builds
/// check this bound before writing anything.
#[inline(always)]
unsafe fn decode(mut bits: u64, base: u32, cursor: *mut u32, end: *const u32) -> *mut u32 {
    let count = bits.count_ones() as usize;
    debug_assert!(
        end as usize - cursor as usize >= count.max(4) * size_of::<u32>(),
        "decode would write past the match buffer"
    );
    // SAFETY: the caller's contract above gives these four slots.
    for slot in 0..4 {
        unsafe { *cursor.add(slot) = base + (bits.trailing_zeros() >> arch::SHIFT) };
        bits &= bits.wrapping_sub(1);
    }
    if bits != 0 {
        let mut slot = 4;
        // SAFETY: one slot per remaining match, still within the caller's `count` slots.
        while bits != 0 {
            unsafe { *cursor.add(slot) = base + (bits.trailing_zeros() >> arch::SHIFT) };
            bits &= bits.wrapping_sub(1);
            slot += 1;
        }
    }
    // SAFETY: `count` slots were just written, so one past the last of them is in bounds.
    unsafe { cursor.add(count) }
}

/// One-pass scan: writes the start offset of every match of `pattern` in `text` into `out` and
/// returns how many. The pattern must not be able to overlap itself, so that every matching
/// position belongs to the non-overlapping match list;
/// [`Literal::matches_into`](crate::literal::Literal::matches_into) routes self-overlapping
/// patterns away.
///
/// The buffer bound is asserted here, not trusted from the caller: everything unsafe in this
/// module leans on it, so the check lives next to what it protects.
pub(crate) fn matches_into<const K: usize>(
    pattern: [u8; K],
    text: &[u8],
    out: &mut [u32],
) -> usize {
    const {
        assert!(
            K >= 1 && K <= 3,
            "the scan is built for patterns of 1 to 3 bytes"
        );
    }
    assert!(
        out.len() >= text.len() / K + 4,
        "the match buffer must hold text.len() / pattern.len() + 4 offsets"
    );
    // SAFETY: the assert above is exactly `scan`'s buffer contract.
    unsafe { scan::<K>(pattern, text, out) }
}

/// # Safety
/// `out.len() >= text.len() / K + 4` must hold; [`matches_into`] asserts it. Every match found
/// counts one slot and matches cannot overlap, so at most `text.len() / K` slots are counted
/// and the four extra keep `decode`'s unconditional writes inside the buffer. Reads from
/// `text` are justified region by region below.
unsafe fn scan<const K: usize>(pattern: [u8; K], text: &[u8], out: &mut [u32]) -> usize {
    let len = text.len();
    if len < K {
        return 0;
    }
    let p = text.as_ptr();
    let mut pattern_chunks = [arch::splat(0); K];
    for (chunk, &byte) in pattern_chunks.iter_mut().zip(&pattern) {
        *chunk = arch::splat(byte);
    }
    let start = out.as_mut_ptr();
    // SAFETY: one past the buffer's last slot is a valid address to form (never read);
    // only `decode`'s debug bound check uses it.
    let end = unsafe { start.add(out.len()) } as *const u32;
    let mut cursor = start;
    let mut i = 0usize;

    // 64 bytes per round; `any` lets a match-free round skip the masks and decodes.
    // SAFETY: the loop condition keeps every read inside `text`: the furthest `match_starts`
    // begins at `p + i + 48` and reads `16 + K - 1` bytes, ending at `i + 64 + (K - 1) <= len`.
    // Writes are `scan`'s buffer contract, handed to `decode` as `end`.
    while i + 64 + (K - 1) <= len {
        let (s0, s1, s2, s3) = unsafe {
            (
                arch::match_starts::<K>(p.add(i), &pattern_chunks),
                arch::match_starts::<K>(p.add(i + 16), &pattern_chunks),
                arch::match_starts::<K>(p.add(i + 32), &pattern_chunks),
                arch::match_starts::<K>(p.add(i + 48), &pattern_chunks),
            )
        };
        if arch::any(s0, s1, s2, s3) {
            unsafe {
                cursor = decode(arch::bits(s0), i as u32, cursor, end);
                cursor = decode(arch::bits(s1), i as u32 + 16, cursor, end);
                cursor = decode(arch::bits(s2), i as u32 + 32, cursor, end);
                cursor = decode(arch::bits(s3), i as u32 + 48, cursor, end);
            }
        }
        i += 64;
    }

    // 16 bytes per round over what the 64-byte rounds left.
    // SAFETY: reads end at `i + 16 + (K - 1) <= len`; writes as above.
    while i + 16 + (K - 1) <= len {
        cursor = unsafe {
            decode(
                arch::bits(arch::match_starts::<K>(p.add(i), &pattern_chunks)),
                i as u32,
                cursor,
                end,
            )
        };
        i += 16;
    }

    if i + K <= len {
        if len >= 16 + (K - 1) {
            // One last 16-byte block, moved back to end flush with the last position a match
            // can still start at. Its low bits re-cover positions the loops above already
            // decoded, so they are masked off.
            // SAFETY: the block reads `base .. base + 16 + K - 1`, which ends exactly at `len`;
            // writes as above.
            let base = len - 16 - (K - 1);
            unsafe {
                let mut bits = arch::bits(arch::match_starts::<K>(p.add(base), &pattern_chunks));
                bits &= !0u64 << ((i - base) << arch::SHIFT);
                cursor = decode(bits, base as u32, cursor, end);
            }
        } else {
            // The whole text is shorter than one block: byte-by-byte costs nothing here.
            while i + K <= len {
                if text[i..i + K] == pattern[..] {
                    debug_assert!(end as usize - cursor as usize >= size_of::<u32>());
                    // SAFETY: one slot per match; the buffer holds a slot for every possible
                    // match plus four ([`matches_into`]'s assert).
                    unsafe {
                        *cursor = i as u32;
                        cursor = cursor.add(1);
                    }
                }
                i += 1;
            }
        }
    }
    // SAFETY: `cursor` and `start` both point into `out`'s buffer.
    unsafe { cursor.offset_from(start) as usize }
}
