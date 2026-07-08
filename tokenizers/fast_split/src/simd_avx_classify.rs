//! x86_64 SIMD classify. Tiered, runtime-detected: AVX-512 VBMI (`vpermb`, native 64-entry) →
//! SSE4.1/SSSE3 128-bit (`pshufb`, covers ~all x86_64 since 2008, incl. every AVX2 CPU) → scalar.
//!
//! Faithful port of the NEON path (`simd_classify.rs`): same tables (`S::tables()`), same algorithm
//! (ASCII fast path → 2-byte group → CJK range → 3-byte peel → tail → MB fixup) — only the lookup
//! primitive differs per ISA, isolated in the `lut*` helpers and selected by the `x86_body!` macro.
//! 16 bytes/iter (128-bit lanes); a 256-/512-bit-wide version is a throughput follow-up.
//!
//! UNTESTED at runtime on aarch64 hosts (cross-compiles only). Validate `== classify_scalar::<S>` on
//! x86_64 (SSE4.1 and, if available, AVX-512 VBMI) hardware before trusting it.
#![allow(unsafe_op_in_unsafe_fn)]

use super::classify::{char_len, classify_scalar, TagScheme};
use core::arch::x86_64::*;

/// Runtime dispatch, best-first. (Swap in memchr's cached `AtomicPtr` if `is_x86_feature_detected!`
/// ever shows up in a profile.)
#[inline]
pub fn dispatch<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    if std::is_x86_feature_detected!("avx512vbmi")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
    {
        unsafe { classify_avx512::<S>(text, tags) } // SAFETY: guarded by the AVX-512 checks
    } else if std::is_x86_feature_detected!("sse4.1") && std::is_x86_feature_detected!("ssse3") {
        unsafe { classify_x86_128::<S>(text, tags) } // SAFETY: guarded by the SSE4.1/SSSE3 checks
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

// ── shared 128-bit primitives (SSE has only signed cmpgt, no byte shift, no 64-entry shuffle) ──
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
unsafe fn uge(a: __m128i, b: u8) -> __m128i {
    _mm_cmpeq_epi8(_mm_max_epu8(a, _mm_set1_epi8(b as i8)), a) // a >= b (unsigned)
}
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
unsafe fn ule(a: __m128i, b: u8) -> __m128i {
    _mm_cmpeq_epi8(_mm_min_epu8(a, _mm_set1_epi8(b as i8)), a) // a <= b (unsigned)
}
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
unsafe fn eqb(a: __m128i, b: u8) -> __m128i {
    _mm_cmpeq_epi8(a, _mm_set1_epi8(b as i8))
}
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
unsafe fn any(m: __m128i) -> bool {
    _mm_movemask_epi8(m) != 0
}
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
unsafe fn hmax(x: __m128i) -> u8 {
    let x = _mm_max_epu8(x, _mm_srli_si128::<8>(x));
    let x = _mm_max_epu8(x, _mm_srli_si128::<4>(x));
    let x = _mm_max_epu8(x, _mm_srli_si128::<2>(x));
    let x = _mm_max_epu8(x, _mm_srli_si128::<1>(x));
    _mm_cvtsi128_si32(x) as u8
}
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
unsafe fn hmin(x: __m128i) -> u8 {
    let x = _mm_min_epu8(x, _mm_srli_si128::<8>(x));
    let x = _mm_min_epu8(x, _mm_srli_si128::<4>(x));
    let x = _mm_min_epu8(x, _mm_srli_si128::<2>(x));
    let x = _mm_min_epu8(x, _mm_srli_si128::<1>(x));
    _mm_cvtsi128_si32(x) as u8
}

// ── SSE4.1 lookups: pshufb is 16-entry, so hi-nibble range mask + OR chains it up ──
/// 128-entry from two 64-byte halves (`idx 0..127 → [lo;hi][idx]`), 8 `pshufb`.
#[target_feature(enable = "ssse3,sse4.1")]
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
/// 256-entry from a contiguous 256-byte table (`idx 0..255 → t[idx]`), 16 `pshufb`.
#[target_feature(enable = "ssse3,sse4.1")]
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

// ── AVX-512 VBMI lookups: vpermb is native 64-entry, so far fewer ops ──
/// 128-entry: 2 `vpermb` (idx&63 into each 64-half) + select by idx bit 6.
#[target_feature(enable = "avx512vbmi,avx512bw,avx512vl")]
#[inline]
unsafe fn lut128_512(lo: &[u8; 64], hi: &[u8; 64], idx: __m128i) -> __m128i {
    let idx512 = _mm512_castsi128_si512(idx);
    let rlo = _mm512_castsi512_si128(_mm512_permutexvar_epi8(idx512, _mm512_loadu_si512(lo.as_ptr() as *const __m512i)));
    let rhi = _mm512_castsi512_si128(_mm512_permutexvar_epi8(idx512, _mm512_loadu_si512(hi.as_ptr() as *const __m512i)));
    let mask = _mm_cmpeq_epi8(_mm_and_si128(idx, _mm_set1_epi8(0x40)), _mm_set1_epi8(0x40)); // idx≥64 → hi
    _mm_blendv_epi8(rlo, rhi, mask)
}
/// 256-entry: 4 `vpermb` (one per 64-slice, idx&63) + select by idx bits 6-7.
#[target_feature(enable = "avx512vbmi,avx512bw,avx512vl")]
#[inline]
unsafe fn lut256_512(t: *const u8, idx: __m128i) -> __m128i {
    let idx512 = _mm512_castsi128_si512(idx);
    let mut acc = _mm_setzero_si128();
    for j in 0..4usize {
        let tj = _mm512_loadu_si512(t.add(j * 64) as *const __m512i);
        let rj = _mm512_castsi512_si128(_mm512_permutexvar_epi8(idx512, tj)); // t[j*64 + (idx&63)]
        let mask = _mm_cmpeq_epi8(_mm_and_si128(idx, _mm_set1_epi8(0xC0u8 as i8)), _mm_set1_epi8((j * 64) as u8 as i8));
        acc = _mm_blendv_epi8(acc, rj, mask);
    }
    acc
}

/// Shared classify body — instantiated by each x86 path with its own `$lut128`/`$lut256`. Everything
/// else (algorithm, other 128-bit ops) is identical. `$S`/`$text`/`$tags` are passed in (macro hygiene).
macro_rules! x86_body {
    ($S:ty, $text:ident, $tags:ident, $lut128:path, $lut256:path) => {{
        let text: &[u8] = $text;
        let tags: &mut [u8] = $tags;
        let (MB, CONT) = (<$S>::MB, <$S>::CONT);
        let tb = <$S>::tables();
        let n = text.len();
        let mut mb_seen = false;
        let mut i = 0usize;
        let z = _mm_setzero_si128();
        let ff = _mm_set1_epi8(0xFFu8 as i8);
        while i + 32 <= n {
            let v = _mm_loadu_si128(text.as_ptr().add(i) as *const __m128i);
            if _mm_movemask_epi8(v) == 0 {
                let out = $lut128(&tb.ascii_lo, &tb.ascii_hi, v);
                _mm_storeu_si128(tags.as_mut_ptr().add(i) as *mut __m128i, out);
                i += 16;
                continue;
            }
            let vn = _mm_loadu_si128(text.as_ptr().add(i + 16) as *const __m128i);
            let b2 = _mm_alignr_epi8::<1>(vn, v);
            let b3 = _mm_alignr_epi8::<2>(vn, v);
            let mut out = $lut128(&tb.ascii_lo, &tb.ascii_hi, v);
            let mut res = z;

            let is2 = _mm_cmpeq_epi8(_mm_and_si128(v, _mm_set1_epi8(0xE0u8 as i8)), _mm_set1_epi8(0xC0u8 as i8));
            if any(is2) {
                let idxg = _mm_or_si128(_mm_slli_epi16::<6>(_mm_and_si128(v, _mm_set1_epi8(3))), _mm_and_si128(b2, _mm_set1_epi8(0x3F)));
                let minl = hmin(_mm_blendv_epi8(ff, v, is2));
                let maxl = hmax(_mm_blendv_epi8(z, v, is2));
                let vsh2 = _mm_and_si128(_mm_srli_epi16::<2>(v), _mm_set1_epi8(0x3F));
                let mut c2 = _mm_set1_epi8(MB as i8);
                let mut g = minl >> 2;
                while g <= (maxl >> 2) {
                    let gg = _mm_and_si128(is2, _mm_cmpeq_epi8(vsh2, _mm_set1_epi8(g as i8)));
                    if any(gg) {
                        let gt = tb.group_tables[(g & 7) as usize].as_ptr() as *const u8;
                        c2 = _mm_blendv_epi8(c2, $lut256(gt, idxg), gg);
                    }
                    g += 1;
                }
                out = _mm_blendv_epi8(out, c2, is2);
                res = is2;
            }

            if let Some(cjk_tag) = <$S>::CJK_RANGE_TAG {
                let iscjk = _mm_and_si128(uge(v, 0xE3), ule(v, 0xED));
                if any(iscjk) {
                    let han = _mm_andnot_si128(_mm_and_si128(eqb(v, 0xE4), eqb(b2, 0xB7)), _mm_and_si128(uge(v, 0xE4), ule(v, 0xE9)));
                    let hg = _mm_or_si128(_mm_or_si128(_mm_and_si128(uge(v, 0xEB), ule(v, 0xEC)), _mm_and_si128(eqb(v, 0xEA), uge(b2, 0xB0))), _mm_and_si128(eqb(v, 0xED), ule(b2, 0x9D)));
                    let e1 = _mm_and_si128(eqb(b2, 0x81), eqb(b3, 0x80));
                    let e2 = _mm_and_si128(eqb(b2, 0x82), _mm_or_si128(_mm_and_si128(uge(b3, 0x97), ule(b3, 0x9C)), eqb(b3, 0xA0)));
                    let e3 = _mm_and_si128(eqb(b2, 0x83), eqb(b3, 0xBB));
                    let kana = _mm_andnot_si128(_mm_or_si128(_mm_or_si128(e1, e2), e3), _mm_and_si128(eqb(v, 0xE3), _mm_and_si128(uge(b2, 0x81), ule(b2, 0x83))));
                    let cjkl = _mm_or_si128(_mm_or_si128(han, hg), kana);
                    out = _mm_blendv_epi8(out, _mm_set1_epi8(cjk_tag as i8), cjkl);
                    res = _mm_or_si128(res, cjkl);
                }
            }

            let is3 = _mm_andnot_si128(res, _mm_and_si128(uge(v, 0xE0), ule(v, 0xEF)));
            if any(is3) {
                let sel = _mm_or_si128(_mm_slli_epi16::<6>(_mm_and_si128(b2, _mm_set1_epi8(1))), _mm_and_si128(b3, _mm_set1_epi8(0x3F)));
                let vp = _mm_and_si128(_mm_srli_epi16::<1>(b2), _mm_set1_epi8(0x7F));
                let mut c3 = _mm_set1_epi8(MB as i8);
                let mut rem = is3;
                while any(rem) {
                    let lead = hmin(_mm_blendv_epi8(ff, v, rem));
                    let ll = _mm_and_si128(rem, eqb(v, lead));
                    let pr = hmin(_mm_blendv_epi8(ff, vp, ll));
                    let gp = _mm_and_si128(ll, _mm_cmpeq_epi8(vp, _mm_set1_epi8(pr as i8)));
                    let idx = (lead - 0xE0) as usize * 32 + (pr & 0x1F) as usize;
                    let k = tb.fast3_uni[idx];
                    let cl = if k != 0xFF {
                        _mm_set1_epi8(k as i8)
                    } else {
                        let (lo, hi) = &tb.fast3_mixed[tb.fast3_slot[idx] as usize];
                        $lut128(lo, hi, sel)
                    };
                    c3 = _mm_blendv_epi8(c3, cl, gp);
                    rem = _mm_andnot_si128(gp, rem);
                }
                out = _mm_blendv_epi8(out, c3, is3);
                res = _mm_or_si128(res, _mm_andnot_si128(eqb(c3, MB), is3));
            }

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
        while i < n {
            let b = text[i];
            if b & 0xC0 == 0x80 {
                tags[i] = CONT;
                i += 1;
                continue;
            }
            tags[i] = <$S>::classify_char(text, i);
            let w = char_len(b);
            let mut j = 1;
            while j < w && i + j < n {
                tags[i + j] = CONT;
                j += 1;
            }
            i += w;
        }
        if mb_seen {
            let mut k = 0;
            while k < n {
                if tags[k] == MB {
                    let cp = decode(text, k);
                    tags[k] = if cp < 0x10000 { tb.bmp_tag(cp as u16) } else { <$S>::classify_char(text, k) };
                }
                k += 1;
            }
        }
    }};
}

/// 128-bit path — SSE4.1/SSSE3 `pshufb`. Runs on ~every x86_64 CPU (≈2008+), incl. all AVX2 machines.
#[target_feature(enable = "ssse3,sse4.1")]
#[allow(non_snake_case)]
unsafe fn classify_x86_128<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    x86_body!(S, text, tags, lut128, lut256)
}

/// AVX-512 VBMI path — `vpermb` native 64-entry lookups (128 = 2 vpermb, 256 = 4), far fewer ops than
/// the `pshufb` chains. Same algorithm/tables. 16 B/iter; full-zmm (64 B/iter) is a further follow-up.
#[target_feature(enable = "avx512vbmi,avx512bw,avx512vl")]
#[allow(non_snake_case)]
unsafe fn classify_avx512<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    x86_body!(S, text, tags, lut128_512, lut256_512)
}
