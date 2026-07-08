//! x86_64 SIMD classify — AVX2 (`vpshufb` = `_mm_shuffle_epi8`), runtime-detected, scalar fallback.
//!
//! Faithful port of the NEON path (`simd_classify.rs`), 16 bytes/iter with 128-bit ops. The DATA is
//! shared — same `S::tables()` drive both — only the lookup primitive differs. NEON `vqtbl4` is a
//! 64-entry shuffle (OOB→0) chained via the subtract trick; `vpshufb` is a **16-entry** shuffle with
//! different OOB rules, so each 64/128/256-entry lookup becomes several `pshufb` + a hi-nibble range
//! mask + OR (`lut128`/`lut256`). Algorithm (ASCII fast path → 2-byte group → CJK range → 3-byte peel
//! → tail → MB fixup) is identical; only the primitives changed.
//!
//! NOTE: 128-bit lanes for a clean port; a 256-bit (`_mm256_shuffle_epi8`, 32 B/iter) version doubles
//! throughput and is the perf follow-up. UNTESTED at runtime on aarch64 hosts — validate
//! `== classify_scalar::<S>` on an x86_64 box before trusting it.
#![allow(unsafe_op_in_unsafe_fn)]

use super::classify::{char_len, classify_scalar, TagScheme};
use core::arch::x86_64::*;

/// Runtime dispatch: AVX2 if present, else the portable scalar walk. (Swap in memchr's cached
/// `AtomicPtr` if `is_x86_feature_detected!` ever shows up in a profile.)
#[inline]
pub fn dispatch<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { classify_avx2::<S>(text, tags) } // SAFETY: guarded by the AVX2 check
    } else {
        classify_scalar::<S>(text, tags)
    }
}

#[inline]
fn decode(t: &[u8], i: usize) -> u32 {
    let b = t[i] as u32;
    match char_len(t[i]) {
        1 => b,
        2 => ((b & 0x1F) << 6) | (t[i + 1] as u32 & 0x3F),
        3 => ((b & 0x0F) << 12) | ((t[i + 1] as u32 & 0x3F) << 6) | (t[i + 2] as u32 & 0x3F),
        _ => {
            ((b & 0x07) << 18)
                | ((t[i + 1] as u32 & 0x3F) << 12)
                | ((t[i + 2] as u32 & 0x3F) << 6)
                | (t[i + 3] as u32 & 0x3F)
        }
    }
}

// ── primitives (SSE has only signed cmpgt, no byte shift, no 64-entry shuffle) ──
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn uge(a: __m128i, b: u8) -> __m128i {
    _mm_cmpeq_epi8(_mm_max_epu8(a, _mm_set1_epi8(b as i8)), a) // a >= b (unsigned)
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ule(a: __m128i, b: u8) -> __m128i {
    _mm_cmpeq_epi8(_mm_min_epu8(a, _mm_set1_epi8(b as i8)), a) // a <= b (unsigned)
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn eqb(a: __m128i, b: u8) -> __m128i {
    _mm_cmpeq_epi8(a, _mm_set1_epi8(b as i8))
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn any(m: __m128i) -> bool {
    _mm_movemask_epi8(m) != 0 // masks are 0xFF/0x00 per lane → any high bit set = any lane matched
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax(x: __m128i) -> u8 {
    let x = _mm_max_epu8(x, _mm_srli_si128::<8>(x));
    let x = _mm_max_epu8(x, _mm_srli_si128::<4>(x));
    let x = _mm_max_epu8(x, _mm_srli_si128::<2>(x));
    let x = _mm_max_epu8(x, _mm_srli_si128::<1>(x));
    _mm_cvtsi128_si32(x) as u8
}
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmin(x: __m128i) -> u8 {
    let x = _mm_min_epu8(x, _mm_srli_si128::<8>(x));
    let x = _mm_min_epu8(x, _mm_srli_si128::<4>(x));
    let x = _mm_min_epu8(x, _mm_srli_si128::<2>(x));
    let x = _mm_min_epu8(x, _mm_srli_si128::<1>(x));
    _mm_cvtsi128_si32(x) as u8
}
/// 128-entry lookup from two 64-byte halves: `idx 0..127 → [lo;hi][idx]`. 8 sub-tables of 16, the
/// hi bits `idx & 0x70` select the sub, `idx & 0x0F` the entry. (idx≥128 lanes → garbage, overwritten.)
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lut128(lo: &[u8; 64], hi: &[u8; 64], idx: __m128i) -> __m128i {
    let idxlo = _mm_and_si128(idx, _mm_set1_epi8(0x0F));
    let sel = _mm_and_si128(idx, _mm_set1_epi8(0x70));
    let mut acc = _mm_setzero_si128();
    for j in 0..8usize {
        let p = if j < 4 { lo.as_ptr().add(j * 16) } else { hi.as_ptr().add((j - 4) * 16) };
        let sub = _mm_loadu_si128(p as *const __m128i);
        let part = _mm_shuffle_epi8(sub, idxlo);
        let m = _mm_cmpeq_epi8(sel, _mm_set1_epi8((j * 16) as i8));
        acc = _mm_or_si128(acc, _mm_and_si128(part, m));
    }
    acc
}
/// 256-entry lookup from a contiguous 256-byte table: `idx 0..255 → t[idx]`. 16 subs, `idx & 0xF0`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lut256(t: *const u8, idx: __m128i) -> __m128i {
    let idxlo = _mm_and_si128(idx, _mm_set1_epi8(0x0F));
    let sel = _mm_and_si128(idx, _mm_set1_epi8(0xF0u8 as i8));
    let mut acc = _mm_setzero_si128();
    for j in 0..16usize {
        let sub = _mm_loadu_si128(t.add(j * 16) as *const __m128i);
        let part = _mm_shuffle_epi8(sub, idxlo);
        let m = _mm_cmpeq_epi8(sel, _mm_set1_epi8((j * 16) as u8 as i8));
        acc = _mm_or_si128(acc, _mm_and_si128(part, m));
    }
    acc
}

/// AVX2 whole-buffer classify, generic over the scheme. Byte-exact target: `classify_scalar::<S>`.
#[target_feature(enable = "avx2")]
#[allow(non_snake_case)]
unsafe fn classify_avx2<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    let (MB, CONT) = (S::MB, S::CONT);
    let tb = S::tables();
    let n = text.len();
    let mut mb_seen = false;
    let mut i = 0;
    let z = _mm_setzero_si128();
    let ff = _mm_set1_epi8(0xFFu8 as i8);

    while i + 32 <= n {
        let v = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);

        // ASCII fast path: no byte has the high bit set → all < 0x80
        if _mm_movemask_epi8(v) == 0 {
            let out = lut128(&tb.ascii_lo, &tb.ascii_hi, v);
            _mm_storeu_si128(tags.as_mut_ptr().add(i) as *mut __m128i, out);
            i += 16;
            continue;
        }

        let vn = _mm_loadu_si128(text.as_ptr().add(i + 16) as *const __m128i);
        let b2 = _mm_alignr_epi8::<1>(vn, v); // each lane i → byte i+1
        let b3 = _mm_alignr_epi8::<2>(vn, v); // each lane i → byte i+2
        let mut out = lut128(&tb.ascii_lo, &tb.ascii_hi, v);
        let mut res = z;

        // 2-byte (C2..DF): loop the lead-group range, lut256 per present group
        let is2 = _mm_cmpeq_epi8(_mm_and_si128(v, _mm_set1_epi8(0xE0u8 as i8)), _mm_set1_epi8(0xC0u8 as i8));
        if any(is2) {
            // idxg = (v&3)<<6 | (b2&0x3F)  — (v&3)≤3 so <<6 has no cross-byte carry
            let idxg = _mm_or_si128(
                _mm_slli_epi16::<6>(_mm_and_si128(v, _mm_set1_epi8(3))),
                _mm_and_si128(b2, _mm_set1_epi8(0x3F)),
            );
            let minl = hmin(_mm_blendv_epi8(ff, v, is2)); // min lead among is2 lanes
            let maxl = hmax(_mm_blendv_epi8(z, v, is2));
            let vsh2 = _mm_and_si128(_mm_srli_epi16::<2>(v), _mm_set1_epi8(0x3F)); // v>>2 (mask the bleed)
            let mut c2 = _mm_set1_epi8(MB as i8);
            let mut g = minl >> 2;
            while g <= (maxl >> 2) {
                let gg = _mm_and_si128(is2, _mm_cmpeq_epi8(vsh2, _mm_set1_epi8(g as i8)));
                if any(gg) {
                    let gt = tb.group_tables[(g & 7) as usize].as_ptr() as *const u8;
                    c2 = _mm_blendv_epi8(c2, lut256(gt, idxg), gg);
                }
                g += 1;
            }
            out = _mm_blendv_epi8(out, c2, is2);
            res = is2;
        }

        // CJK (E3..ED) range shortcut — only when the scheme maps all of CJK to one tag
        if let Some(cjk_tag) = S::CJK_RANGE_TAG {
            let iscjk = _mm_and_si128(uge(v, 0xE3), ule(v, 0xED));
            if any(iscjk) {
                let han = _mm_andnot_si128(
                    _mm_and_si128(eqb(v, 0xE4), eqb(b2, 0xB7)),
                    _mm_and_si128(uge(v, 0xE4), ule(v, 0xE9)),
                );
                let hg = _mm_or_si128(
                    _mm_or_si128(
                        _mm_and_si128(uge(v, 0xEB), ule(v, 0xEC)),
                        _mm_and_si128(eqb(v, 0xEA), uge(b2, 0xB0)),
                    ),
                    _mm_and_si128(eqb(v, 0xED), ule(b2, 0x9D)),
                );
                let e1 = _mm_and_si128(eqb(b2, 0x81), eqb(b3, 0x80));
                let e2 = _mm_and_si128(
                    eqb(b2, 0x82),
                    _mm_or_si128(_mm_and_si128(uge(b3, 0x97), ule(b3, 0x9C)), eqb(b3, 0xA0)),
                );
                let e3 = _mm_and_si128(eqb(b2, 0x83), eqb(b3, 0xBB));
                let kana = _mm_andnot_si128(
                    _mm_or_si128(_mm_or_si128(e1, e2), e3),
                    _mm_and_si128(eqb(v, 0xE3), _mm_and_si128(uge(b2, 0x81), ule(b2, 0x83))),
                );
                let cjkl = _mm_or_si128(_mm_or_si128(han, hg), kana);
                out = _mm_blendv_epi8(out, _mm_set1_epi8(cjk_tag as i8), cjkl);
                res = _mm_or_si128(res, cjkl);
            }
        }

        // 3-byte non-CJK: exact peel of the distinct (lead, b2-pair) blocks present
        let is3 = _mm_andnot_si128(res, _mm_and_si128(uge(v, 0xE0), ule(v, 0xEF)));
        if any(is3) {
            let sel = _mm_or_si128(
                _mm_slli_epi16::<6>(_mm_and_si128(b2, _mm_set1_epi8(1))),
                _mm_and_si128(b3, _mm_set1_epi8(0x3F)),
            );
            let vp = _mm_and_si128(_mm_srli_epi16::<1>(b2), _mm_set1_epi8(0x7F)); // b2>>1 (pair id)
            let mut c3 = _mm_set1_epi8(MB as i8);
            let mut rem = is3;
            while any(rem) {
                let lead = hmin(_mm_blendv_epi8(ff, v, rem)); // min lead among unresolved
                let ll = _mm_and_si128(rem, eqb(v, lead));
                let pr = hmin(_mm_blendv_epi8(ff, vp, ll)); // min b2-pair within that lead
                let gp = _mm_and_si128(ll, _mm_cmpeq_epi8(vp, _mm_set1_epi8(pr as i8)));
                let idx = (lead - 0xE0) as usize * 32 + (pr & 0x1F) as usize;
                let k = tb.fast3_uni[idx];
                let cl = if k != 0xFF {
                    _mm_set1_epi8(k as i8) // uniform block: one constant
                } else {
                    let (lo, hi) = &tb.fast3_mixed[tb.fast3_slot[idx] as usize];
                    lut128(lo, hi, sel)
                };
                c3 = _mm_blendv_epi8(c3, cl, gp);
                rem = _mm_andnot_si128(gp, rem); // rem &= ~gp
            }
            out = _mm_blendv_epi8(out, c3, is3);
            res = _mm_or_si128(res, _mm_andnot_si128(eqb(c3, MB), is3));
        }

        // residual multibyte lead → MB ; continuation byte → CONT
        let leadmb = _mm_andnot_si128(res, uge(v, 0xC0));
        out = _mm_blendv_epi8(out, _mm_set1_epi8(MB as i8), leadmb);
        let cont = _mm_cmpeq_epi8(_mm_and_si128(v, _mm_set1_epi8(0xC0u8 as i8)), _mm_set1_epi8(0x80u8 as i8));
        out = _mm_blendv_epi8(out, _mm_set1_epi8(CONT as i8), cont);

        if any(eqb(out, MB)) {
            mb_seen = true;
        }
        _mm_storeu_si128(tags.as_mut_ptr().add(i) as *mut __m128i, out);
        i += 16;
    }

    // scalar tail (< 32 bytes)
    while i < n {
        let b = text[i];
        if b & 0xC0 == 0x80 {
            tags[i] = CONT;
            i += 1;
            continue;
        }
        tags[i] = S::classify_char(text, i);
        let w = char_len(b);
        let mut j = 1;
        while j < w && i + j < n {
            tags[i + j] = CONT;
            j += 1;
        }
        i += w;
    }

    // MB fixup: resolve every lane the SIMD left as MB (CJK holes, astral)
    if mb_seen {
        let mut k = 0;
        while k < n {
            if tags[k] == MB {
                let cp = decode(text, k);
                tags[k] = if cp < 0x10000 { tb.bmp_tag(cp as u16) } else { S::classify_char(text, k) };
            }
            k += 1;
        }
    }
}
