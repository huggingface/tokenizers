//! wasm32 SIMD128 classify (`u8x16_swizzle`). Compiled only when the build enables `simd128`
//! (WASM has no runtime feature detection — it's all-or-nothing at compile time); otherwise the
//! dispatcher in `classify.rs` uses the scalar walk.
//!
//! The closest of all the ports: `u8x16_swizzle` is a 16-entry shuffle with **OOB→0** — exactly NEON
//! `vqtbl`'s semantics — so the subtract trick carries over directly (no hi-nibble range masks like
//! x86 needs). WASM also has native unsigned compares (`u8x16_ge/le`) and per-lane byte shifts
//! (`i8x16_shl`/`u8x16_shr`, no cross-byte bleed), so the body is a near 1:1 map of the NEON path.
//! Same tables (`S::tables()`), same algorithm. 16 bytes/iter.
//!
//! UNTESTED at runtime on aarch64 hosts (cross-compiles only). Validate `== classify_scalar::<S>` in
//! a SIMD128 wasm engine before trusting it.
#![allow(unsafe_op_in_unsafe_fn)]

use super::classify::{TagScheme, char_len};
use core::arch::wasm32::*;

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

#[inline]
unsafe fn eqb(a: v128, b: u8) -> v128 {
    u8x16_eq(a, u8x16_splat(b))
}
#[inline]
unsafe fn uge(a: v128, b: u8) -> v128 {
    u8x16_ge(a, u8x16_splat(b))
}
#[inline]
unsafe fn ule(a: v128, b: u8) -> v128 {
    u8x16_le(a, u8x16_splat(b))
}
#[inline]
unsafe fn hmax(x: v128) -> u8 {
    let x = u8x16_max(
        x,
        u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15>(x, x),
    );
    let x = u8x16_max(
        x,
        u8x16_shuffle::<4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7>(x, x),
    );
    let x = u8x16_max(
        x,
        u8x16_shuffle::<2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3>(x, x),
    );
    let x = u8x16_max(
        x,
        u8x16_shuffle::<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(x, x),
    );
    u8x16_extract_lane::<0>(x)
}
#[inline]
unsafe fn hmin(x: v128) -> u8 {
    let x = u8x16_min(
        x,
        u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15>(x, x),
    );
    let x = u8x16_min(
        x,
        u8x16_shuffle::<4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7>(x, x),
    );
    let x = u8x16_min(
        x,
        u8x16_shuffle::<2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3>(x, x),
    );
    let x = u8x16_min(
        x,
        u8x16_shuffle::<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(x, x),
    );
    u8x16_extract_lane::<0>(x)
}
/// 128-entry from two 64-byte halves via 8× 16-entry swizzle + subtract trick (OOB→0 does the range
/// gating — no explicit mask). (idx≥128 → all sub-swizzles OOB → 0, like NEON; overwritten later.)
#[inline]
unsafe fn lut128_w(lo: &[u8; 64], hi: &[u8; 64], idx: v128) -> v128 {
    let mut acc = u8x16_splat(0);
    for j in 0..8usize {
        let p = if j < 4 {
            lo.as_ptr().add(j * 16)
        } else {
            hi.as_ptr().add((j - 4) * 16)
        };
        let t = v128_load(p as *const v128);
        acc = v128_or(
            acc,
            u8x16_swizzle(t, i8x16_sub(idx, u8x16_splat((j * 16) as u8))),
        );
    }
    acc
}
/// 256-entry from a contiguous 256-byte table via 16× swizzle + subtract.
#[inline]
unsafe fn lut256_w(t: *const u8, idx: v128) -> v128 {
    let mut acc = u8x16_splat(0);
    for j in 0..16usize {
        let tt = v128_load(t.add(j * 16) as *const v128);
        acc = v128_or(
            acc,
            u8x16_swizzle(tt, i8x16_sub(idx, u8x16_splat((j * 16) as u8))),
        );
    }
    acc
}

/// SIMD128 whole-buffer classify, generic over the scheme. Byte-exact target: `classify_scalar::<S>`.
#[allow(non_snake_case)]
pub unsafe fn classify_wasm<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    let (MB, CONT) = (S::MB, S::CONT);
    let tb = S::tables();
    let n = text.len();
    let mut mb_seen = false;
    let mut i = 0usize;
    let ff = u8x16_splat(0xFF);
    let zero = u8x16_splat(0);

    while i + 32 <= n {
        let v = v128_load(text.as_ptr().add(i) as *const v128);

        // ASCII fast path: no lane has the high bit set
        if u8x16_bitmask(v) == 0 {
            let out = lut128_w(&tb.ascii_lo, &tb.ascii_hi, v);
            v128_store(tags.as_mut_ptr().add(i) as *mut v128, out);
            i += 16;
            continue;
        }

        let vn = v128_load(text.as_ptr().add(i + 16) as *const v128);
        let b2 = u8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(v, vn); // byte i+1
        let b3 = u8x16_shuffle::<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17>(v, vn); // byte i+2
        let mut out = lut128_w(&tb.ascii_lo, &tb.ascii_hi, v);
        let mut res = zero;

        // 2-byte (C2..DF): loop the lead-group range, lut256 per present group
        let is2 = u8x16_eq(v128_and(v, u8x16_splat(0xE0)), u8x16_splat(0xC0));
        if v128_any_true(is2) {
            let idxg = v128_or(
                i8x16_shl(v128_and(v, u8x16_splat(3)), 6),
                v128_and(b2, u8x16_splat(0x3F)),
            );
            let minl = hmin(v128_bitselect(v, ff, is2)); // v where is2 else 0xFF
            let maxl = hmax(v128_bitselect(v, zero, is2));
            let vsh2 = u8x16_shr(v, 2);
            let mut c2 = u8x16_splat(MB);
            let mut g = minl >> 2;
            while g <= (maxl >> 2) {
                let gg = v128_and(is2, u8x16_eq(vsh2, u8x16_splat(g)));
                if v128_any_true(gg) {
                    let gt = tb.group_tables[(g & 7) as usize].as_ptr() as *const u8;
                    c2 = v128_bitselect(lut256_w(gt, idxg), c2, gg);
                }
                g += 1;
            }
            out = v128_bitselect(c2, out, is2);
            res = is2;
        }

        // CJK (E3..ED) range shortcut — only when the scheme maps all of CJK to one tag
        if let Some(cjk_tag) = S::CJK_RANGE_TAG {
            let iscjk = v128_and(uge(v, 0xE3), ule(v, 0xED));
            if v128_any_true(iscjk) {
                let han = v128_andnot(
                    v128_and(uge(v, 0xE4), ule(v, 0xE9)),
                    v128_and(eqb(v, 0xE4), eqb(b2, 0xB7)),
                );
                let hg = v128_or(
                    v128_or(
                        v128_and(uge(v, 0xEB), ule(v, 0xEC)),
                        v128_and(eqb(v, 0xEA), uge(b2, 0xB0)),
                    ),
                    v128_and(eqb(v, 0xED), ule(b2, 0x9D)),
                );
                let e1 = v128_and(eqb(b2, 0x81), eqb(b3, 0x80));
                let e2 = v128_and(
                    eqb(b2, 0x82),
                    v128_or(v128_and(uge(b3, 0x97), ule(b3, 0x9C)), eqb(b3, 0xA0)),
                );
                let e3 = v128_and(eqb(b2, 0x83), eqb(b3, 0xBB));
                let kana = v128_andnot(
                    v128_and(eqb(v, 0xE3), v128_and(uge(b2, 0x81), ule(b2, 0x83))),
                    v128_or(v128_or(e1, e2), e3),
                );
                let cjkl = v128_or(v128_or(han, hg), kana);
                out = v128_bitselect(u8x16_splat(cjk_tag), out, cjkl);
                res = v128_or(res, cjkl);
            }
        }

        // 3-byte non-CJK: exact peel of the distinct (lead, b2-pair) blocks present
        let is3 = v128_andnot(v128_and(uge(v, 0xE0), ule(v, 0xEF)), res);
        if v128_any_true(is3) {
            let sel = v128_or(
                i8x16_shl(v128_and(b2, u8x16_splat(1)), 6),
                v128_and(b3, u8x16_splat(0x3F)),
            );
            let vp = u8x16_shr(b2, 1); // b2>>1 (pair id)
            let mut c3 = u8x16_splat(MB);
            let mut rem = is3;
            while v128_any_true(rem) {
                let lead = hmin(v128_bitselect(v, ff, rem));
                let ll = v128_and(rem, eqb(v, lead));
                let pr = hmin(v128_bitselect(vp, ff, ll));
                let gp = v128_and(ll, u8x16_eq(vp, u8x16_splat(pr)));
                let idx = (lead - 0xE0) as usize * 32 + (pr & 0x1F) as usize;
                let k = tb.fast3_uni[idx];
                let cl = if k != 0xFF {
                    u8x16_splat(k)
                } else {
                    let (lo, hi) = &tb.fast3_mixed[tb.fast3_slot[idx] as usize];
                    lut128_w(lo, hi, sel)
                };
                c3 = v128_bitselect(cl, c3, gp);
                rem = v128_andnot(rem, gp); // rem &= ~gp
            }
            out = v128_bitselect(c3, out, is3);
            res = v128_or(res, v128_andnot(is3, eqb(c3, MB)));
        }

        // residual multibyte lead → MB ; continuation byte → CONT
        let leadmb = v128_andnot(uge(v, 0xC0), res);
        out = v128_bitselect(u8x16_splat(MB), out, leadmb);
        let cont = u8x16_eq(v128_and(v, u8x16_splat(0xC0)), u8x16_splat(0x80));
        out = v128_bitselect(u8x16_splat(CONT), out, cont);

        if v128_any_true(eqb(out, MB)) {
            mb_seen = true;
        }
        v128_store(tags.as_mut_ptr().add(i) as *mut v128, out);
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

    // MB fixup
    if mb_seen {
        let mut k = 0;
        while k < n {
            if tags[k] == MB {
                let cp = decode(text, k);
                tags[k] = if cp < 0x10000 {
                    tb.bmp_tag(cp as u16)
                } else {
                    S::classify_char(text, k)
                };
            }
            k += 1;
        }
    }
}
