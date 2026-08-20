//! SSSE3 block builder: the x86 twin of `simd.rs`. Same shape — tags through a LUT into dense
//! codes, continuation lanes filled from the left, three bit-planes extracted — with two
//! differences that fall out of the ISA.
//!
//! Plane extraction is cheaper here: `movemask` is native, so a plane is
//! `movemask(slli_epi16(code, 7 - bit))`. Shifting 16-bit lanes puts bit `k` of the low byte at
//! bit 7 and bit `k` of the high byte at bit 15, which is exactly the pair `movemask` reads; the
//! bits that bleed across the byte boundary land where it does not look.
//!
//! The LUT is dearer: `pshufb` indexes 16 entries and the tags run to 0x26, so the table is split
//! into one shuffle per high nibble and selected between. Refinements (`0x10`/`0x20` case, `0x16`
//! AlphaSymMark, `0x26` ZWJ) are the only tags above 0x0F, so three shuffles cover every case.
//!
//! Checked the same way as the NEON kernel: `src/bin/verify.rs` built for `x86_64-apple-darwin`
//! and run under Rosetta, which reports SSSE3, so this path is the one that executes. The scalar
//! builder in `lib.rs` stays the reference it is compared against.

use crate::{AUX_CJK, AUX_NONE, AUX_SLASH, Blk, lead_run};
use core::arch::x86_64::*;

/// `x <= k`, unsigned, in the absence of an unsigned byte compare.
#[inline(always)]
unsafe fn le(x: __m128i, k: u8) -> __m128i {
    unsafe { _mm_cmpeq_epi8(_mm_min_epu8(x, _mm_set1_epi8(k as i8)), x) }
}

/// `x >= k`, unsigned.
#[inline(always)]
unsafe fn ge(x: __m128i, k: u8) -> __m128i {
    unsafe { _mm_cmpeq_epi8(_mm_max_epu8(x, _mm_set1_epi8(k as i8)), x) }
}

/// `x - lo <= n`, unsigned, i.e. `x` in `lo ..= lo + n`.
#[inline(always)]
unsafe fn in_range(x: __m128i, lo: u8, n: u8) -> __m128i {
    unsafe { le(_mm_sub_epi8(x, _mm_set1_epi8(lo as i8)), n) }
}

/// `if mask { a } else { b }`, lane-wise. SSE4.1's `blendv` would do it in one, but the and/andnot
/// pair keeps the whole builder at SSSE3.
#[inline(always)]
unsafe fn sel(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe { _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b)) }
}

/// 64 bytes of `text[base..]` compared against `b`, as a bitmap.
///
/// # Safety
/// `base + 64 <= text.len()`; the caller has checked for SSSE3.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn eq64(text: &[u8], base: usize, b: u8) -> u64 {
    unsafe {
        let n = _mm_set1_epi8(b as i8);
        let mut m = 0u64;
        for k in 0..4 {
            let v = _mm_loadu_si128(text.as_ptr().add(base + k * 16).cast());
            m |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, n)) as u16 as u64) << (16 * k);
        }
        m
    }
}

/// Build one **full** 64-byte block. `cur_code`/`cur_aux` describe the byte before it.
///
/// # Safety
/// `base + 64 <= tags.len()` and `base + 64 <= text.len()`; the caller has checked for SSSE3.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn build64<const AUX: u8, const P3: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    lut: &[u8; 64],
    cur_code: u8,
    cur_aux: bool,
) -> (Blk, u8) {
    unsafe {
        let low_nibble = _mm_set1_epi8(0x0F);
        let seven = _mm_set1_epi8(7);
        // one table per high nibble; a tag of 0x30 or above never occurs
        let t0 = _mm_loadu_si128(lut.as_ptr().cast());
        let t1 = _mm_loadu_si128(lut.as_ptr().add(16).cast());
        let t2 = _mm_loadu_si128(lut.as_ptr().add(32).cast());

        let mut cd = [_mm_setzero_si128(); 4];
        let mut isc = [_mm_setzero_si128(); 4];
        let mut prev = _mm_set1_epi8(cur_code as i8);
        for k in 0..4 {
            let t = _mm_loadu_si128(tags.as_ptr().add(base + k * 16).cast());
            let lo = _mm_and_si128(t, low_nibble);
            let hi = _mm_and_si128(_mm_srli_epi16(t, 4), low_nibble);
            let raw = sel(
                _mm_cmpeq_epi8(hi, _mm_set1_epi8(2)),
                _mm_shuffle_epi8(t2, lo),
                sel(
                    _mm_cmpeq_epi8(hi, _mm_set1_epi8(1)),
                    _mm_shuffle_epi8(t1, lo),
                    _mm_shuffle_epi8(t0, lo),
                ),
            );
            isc[k] = _mm_cmpeq_epi8(raw, seven);
            // fill the continuation lanes: shift by one, then shift *that* by two, which covers
            // the three continuations a 4-byte char can have.
            let c = sel(isc[k], _mm_alignr_epi8(raw, prev, 15), raw);
            let c = sel(_mm_cmpeq_epi8(c, seven), _mm_alignr_epi8(c, prev, 14), c);
            prev = c;
            cd[k] = c;
        }
        let last_code = (_mm_extract_epi16(prev, 7) >> 8) as u8;

        // 64 lanes -> one u64, four native movemasks
        let gather = |v: [__m128i; 4]| -> u64 {
            let mut m = 0u64;
            for (k, chunk) in v.iter().enumerate() {
                m |= (_mm_movemask_epi8(*chunk) as u16 as u64) << (16 * k);
            }
            m
        };
        // the shift is a const generic, so one arm per plane: bit k wants `7 - k`
        macro_rules! plane {
            ($shift:literal) => {
                gather([
                    _mm_slli_epi16::<$shift>(cd[0]),
                    _mm_slli_epi16::<$shift>(cd[1]),
                    _mm_slli_epi16::<$shift>(cd[2]),
                    _mm_slli_epi16::<$shift>(cd[3]),
                ])
            };
        }

        let mut b = Blk {
            cont: gather(isc),
            p0: plane!(7),
            p1: plane!(6),
            p2: plane!(5),
            p3: if P3 { plane!(4) } else { 0 },
            aux: 0,
        };
        if AUX == AUX_NONE {
            return (b, last_code);
        }

        if AUX == AUX_SLASH {
            // single ASCII byte — no fill needed, `/` is its own char
            let sl = _mm_set1_epi8(b'/' as i8);
            b.aux = gather([
                _mm_cmpeq_epi8(_mm_loadu_si128(text.as_ptr().add(base).cast()), sl),
                _mm_cmpeq_epi8(_mm_loadu_si128(text.as_ptr().add(base + 16).cast()), sl),
                _mm_cmpeq_epi8(_mm_loadu_si128(text.as_ptr().add(base + 32).cast()), sl),
                _mm_cmpeq_epi8(_mm_loadu_si128(text.as_ptr().add(base + 48).cast()), sl),
            ]);
            return (b, last_code);
        }

        // ── the CJK range test, on the raw bytes: Hiragana/Katakana U+3040..30FF is E3 [81-83] xx;
        // Han U+4E00..9FA5 is E4 [B8-BF] xx / E5-E8 xx xx / E9 [80-BD] xx / E9 BE [80-A5].
        let ntext = text.len();
        let tv = [
            _mm_loadu_si128(text.as_ptr().add(base).cast()),
            _mm_loadu_si128(text.as_ptr().add(base + 16).cast()),
            _mm_loadu_si128(text.as_ptr().add(base + 32).cast()),
            _mm_loadu_si128(text.as_ptr().add(base + 48).cast()),
        ];
        let any = tv
            .iter()
            .any(|v| _mm_movemask_epi8(in_range(*v, 0xE3, 6)) != 0);
        if !any {
            if cur_aux {
                b.aux |= lead_run(b.cont, !0);
            }
            return (b, last_code);
        }
        let tail = {
            let mut buf = [0u8; 16];
            let off = base + 64;
            let avail = ntext.saturating_sub(off).min(16);
            buf[..avail].copy_from_slice(&text[off..off + avail]);
            _mm_loadu_si128(buf.as_ptr().cast())
        };
        let cjkv = |v: __m128i, nx: __m128i| -> __m128i {
            let b1 = _mm_alignr_epi8(nx, v, 1);
            let b2 = _mm_alignr_epi8(nx, v, 2);
            let e9 = _mm_and_si128(
                _mm_cmpeq_epi8(v, _mm_set1_epi8(0xE9u8 as i8)),
                _mm_or_si128(
                    le(b1, 0xBD),
                    _mm_and_si128(
                        _mm_cmpeq_epi8(b1, _mm_set1_epi8(0xBEu8 as i8)),
                        le(b2, 0xA5),
                    ),
                ),
            );
            _mm_or_si128(
                _mm_or_si128(
                    _mm_and_si128(
                        _mm_cmpeq_epi8(v, _mm_set1_epi8(0xE3u8 as i8)),
                        in_range(b1, 0x81, 2),
                    ),
                    _mm_and_si128(_mm_cmpeq_epi8(v, _mm_set1_epi8(0xE4u8 as i8)), ge(b1, 0xB8)),
                ),
                _mm_or_si128(in_range(v, 0xE5, 3), e9),
            )
        };
        let mut leads = gather([
            cjkv(tv[0], tv[1]),
            cjkv(tv[1], tv[2]),
            cjkv(tv[2], tv[3]),
            cjkv(tv[3], tail),
        ]);
        // `fsm_deepseek` reads three bytes unconditionally; refuse to classify a truncated tail.
        let lim = ntext.saturating_sub(base + 2);
        if lim < 64 {
            leads &= (1u64 << lim) - 1;
        }
        // every CJK char is 3 bytes, so two shifts fill it; one cut by the block edge is picked up
        // on the other side by `cur_aux`.
        b.aux = leads | (leads << 1) | (leads << 2);
        if cur_aux {
            b.aux |= lead_run(b.cont, !0);
        }
        (b, last_code)
    }
}
