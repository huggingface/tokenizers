//! wasm32 SIMD128 classify (`u8x16_swizzle`). Compiled only when the build enables `simd128`
//! (WASM has no runtime feature detection — it's all-or-nothing at compile time); otherwise the
//! dispatcher in `classify.rs` uses the scalar walk.
//!
//! The closest of all the ports: `u8x16_swizzle` is a 16-entry shuffle with **OOB→0** — exactly NEON
//! `vqtbl`'s semantics — so the subtract trick carries over directly (no hi-nibble range masks like
//! x86 needs). WASM also has native unsigned compares (`u8x16_ge/le`) and per-lane byte shifts
//! (`i8x16_shl`/`u8x16_shr`, no cross-byte bleed), so the body is a near 1:1 map of the NEON path.
//! Same tables (`ATOM_TABLES`), same algorithm. 16 bytes/iter.
//!
//! Per lane, `b0`/`b1`/`b2` are the 1st/2nd/3rd bytes of the (potential) UTF-8 char starting there —
//! i.e. the byte at the lane and the two after it (built with cross-chunk shuffles).
//!
//! UNTESTED at runtime on aarch64 hosts (cross-compiles only). Validate `== classify_scalar` in
//! a SIMD128 wasm engine before trusting it.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::atom_tables::ATOM_TABLES;
use crate::classify::{Atom, char_len, CONT, MB};
use core::arch::wasm32::*;

const CJK_TAG: u8 = Atom::Letter as u8;

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

// ── lane predicates (each returns an all-ones / all-zeros mask per lane) ──
/// Lanes equal to `byte`.
#[inline(always)]
unsafe fn eq(bytes: v128, byte: u8) -> v128 {
    u8x16_eq(bytes, u8x16_splat(byte))
}
/// Lanes `>= lo` (unsigned).
#[inline(always)]
unsafe fn ge(bytes: v128, lo: u8) -> v128 {
    u8x16_ge(bytes, u8x16_splat(lo))
}
/// Lanes `<= hi` (unsigned).
#[inline(always)]
unsafe fn le(bytes: v128, hi: u8) -> v128 {
    u8x16_le(bytes, u8x16_splat(hi))
}
/// Lanes with `lo <= byte <= hi` (unsigned, inclusive) — the readable form of the range compares.
#[inline(always)]
unsafe fn in_range(bytes: v128, lo: u8, hi: u8) -> v128 {
    v128_and(ge(bytes, lo), le(bytes, hi))
}

/// Horizontal max over the 16 lanes (shuffle-fold; duplicated lanes never raise the max).
#[inline(always)]
unsafe fn hmax(v: v128) -> u8 {
    let v = u8x16_max(
        v,
        u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15>(v, v),
    );
    let v = u8x16_max(
        v,
        u8x16_shuffle::<4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7>(v, v),
    );
    let v = u8x16_max(
        v,
        u8x16_shuffle::<2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3>(v, v),
    );
    let v = u8x16_max(
        v,
        u8x16_shuffle::<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(v, v),
    );
    u8x16_extract_lane::<0>(v)
}
/// Horizontal min over the 16 lanes (fill excluded lanes with 0xFF before calling so they don't win).
#[inline(always)]
unsafe fn hmin(v: v128) -> u8 {
    let v = u8x16_min(
        v,
        u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15>(v, v),
    );
    let v = u8x16_min(
        v,
        u8x16_shuffle::<4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7>(v, v),
    );
    let v = u8x16_min(
        v,
        u8x16_shuffle::<2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3>(v, v),
    );
    let v = u8x16_min(
        v,
        u8x16_shuffle::<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(v, v),
    );
    u8x16_extract_lane::<0>(v)
}

/// 128-entry lookup from two 64-byte halves via 8× 16-entry swizzle + subtract trick (OOB→0 does the
/// range gating — no explicit mask). (index≥128 → every sub-swizzle OOB → 0, like NEON; overwritten.)
#[inline(always)]
unsafe fn lookup128(lo: &[u8; 64], hi: &[u8; 64], index: v128) -> v128 {
    let mut acc = u8x16_splat(0);
    for sub in 0..8usize {
        let base = if sub < 4 {
            lo.as_ptr().add(sub * 16)
        } else {
            hi.as_ptr().add((sub - 4) * 16)
        };
        let table = v128_load(base as *const v128);
        acc = v128_or(
            acc,
            u8x16_swizzle(table, i8x16_sub(index, u8x16_splat((sub * 16) as u8))),
        );
    }
    acc
}
/// 256-entry lookup from a contiguous 256-byte table via 16× swizzle + subtract.
#[inline(always)]
unsafe fn lookup256(table: *const u8, index: v128) -> v128 {
    let mut acc = u8x16_splat(0);
    for sub in 0..16usize {
        let chunk = v128_load(table.add(sub * 16) as *const v128);
        acc = v128_or(
            acc,
            u8x16_swizzle(chunk, i8x16_sub(index, u8x16_splat((sub * 16) as u8))),
        );
    }
    acc
}

/// SIMD128 whole-buffer classify. Byte-exact target: `classify_scalar`.
///
/// # Safety
/// `tags.len()` must be ≥ `text.len()` — the kernel does raw 16-byte `v128_store`s into `tags` for full
/// chunks. `text` must be well-formed UTF-8 (the tail/astral scalar path reads a lead's continuation
/// bytes). Both hold when called via [`crate::classify`], which asserts the length up front.
pub unsafe fn classify_wasm(text: &[u8], tags: &mut [u8]) {
    let n = text.len();
    let mut mb_seen = false;
    let mut i = 0usize;
    let ones = u8x16_splat(0xFF);
    let zeros = u8x16_splat(0);

    while i + 32 <= n {
        let b0 = v128_load(text.as_ptr().add(i) as *const v128);

        // ASCII fast path: no lane has the high bit set
        if u8x16_bitmask(b0) == 0 {
            let out = lookup128(&ATOM_TABLES.ascii_lo, &ATOM_TABLES.ascii_hi, b0);
            v128_store(tags.as_mut_ptr().add(i) as *mut v128, out);
            i += 16;
            continue;
        }

        let next = v128_load(text.as_ptr().add(i + 16) as *const v128);
        let b1 = u8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(b0, next); // byte at lane+1
        let b2 = u8x16_shuffle::<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17>(b0, next); // byte at lane+2
        let mut out = lookup128(&ATOM_TABLES.ascii_lo, &ATOM_TABLES.ascii_hi, b0);
        let mut resolved = zeros;

        // 2-byte (C2..DF, i.e. lead & 0xE0 == 0xC0): loop the lead-group range, lookup256 per present group
        let is_lead2 = u8x16_eq(v128_and(b0, u8x16_splat(0xE0)), u8x16_splat(0xC0));
        if v128_any_true(is_lead2) {
            let group_index = v128_or(
                i8x16_shl(v128_and(b0, u8x16_splat(3)), 6),
                v128_and(b1, u8x16_splat(0x3F)),
            );
            let min_lead = hmin(v128_bitselect(b0, ones, is_lead2)); // lead where is_lead2 else 0xFF
            let max_lead = hmax(v128_bitselect(b0, zeros, is_lead2));
            let lead_group = u8x16_shr(b0, 2); // which of the 8 lead-groups each lane's lead belongs to
            let mut tags2 = u8x16_splat(MB);
            let mut group = min_lead >> 2;
            while group <= (max_lead >> 2) {
                let this_group = v128_and(is_lead2, u8x16_eq(lead_group, u8x16_splat(group)));
                if v128_any_true(this_group) {
                    let group_table =
                        ATOM_TABLES.group_tables[(group & 7) as usize].as_ptr() as *const u8;
                    tags2 = v128_bitselect(lookup256(group_table, group_index), tags2, this_group);
                }
                group += 1;
            }
            out = v128_bitselect(tags2, out, is_lead2);
            resolved = is_lead2;
        }

        // CJK fast path (leads E3..ED = U+3000..U+DFFF). Enabled only when the scheme collapses all of
        // CJK to a single tag (Atoms → Letter). This is the OPTIMISTIC bulk: it flags only lanes that
        // are DEFINITELY that tag; boundary/hole codepoints it leaves unresolved, so they fall through
        // to the exact 3-byte tables below. It never over-claims, so the result stays byte-exact.
        let in_cjk_leads = in_range(b0, 0xE3, 0xED);
        if v128_any_true(in_cjk_leads) {
            // Han — U+4000..U+9FFF (CJK Unified Ideographs + the Ext-A tail), minus the one
            // non-ideograph hole U+4DC0..U+4DFF (Yijing Hexagram Symbols), which encodes as E4 B7 xx.
            let han = v128_andnot(
                in_range(b0, 0xE4, 0xE9),
                v128_and(eq(b0, 0xE4), eq(b1, 0xB7)),
            );

            // Hangul Syllables (U+AC00..U+D7A3), split across leads EA..ED:
            //   EB..EC        → U+B000..U+CFFF  (whole middle — every lane a syllable)
            //   EA, b1 >= B0  → U+AC00..U+AFFF  (syllables begin at AC00; U+A000..ABFF below is excluded)
            //   ED, b1 <= 9D  → U+D000..U+D77F  (syllables; the U+D780.. tail is left to the exact tables)
            let hangul = v128_or(
                v128_or(
                    in_range(b0, 0xEB, 0xEC),
                    v128_and(eq(b0, 0xEA), ge(b1, 0xB0)),
                ),
                v128_and(eq(b0, 0xED), le(b1, 0x9D)),
            );

            // Kana — Hiragana + Katakana U+3040..U+30FF (lead E3, b1 in 81..83), minus the
            // non-letter holes inside that block:
            //   U+3040          reserved                 (E3 81 80)
            //   U+3097..U+309C  unassigned + combining    (E3 82 97..9C)
            //   U+30A0          double hyphen (Punct)      (E3 82 A0)
            //   U+30FB          middle dot (Punct)         (E3 83 BB)
            let hole_3040 = v128_and(eq(b1, 0x81), eq(b2, 0x80));
            let hole_309x = v128_and(
                eq(b1, 0x82),
                v128_or(in_range(b2, 0x97, 0x9C), eq(b2, 0xA0)),
            );
            let hole_30fb = v128_and(eq(b1, 0x83), eq(b2, 0xBB));
            let kana = v128_andnot(
                v128_and(eq(b0, 0xE3), in_range(b1, 0x81, 0x83)),
                v128_or(v128_or(hole_3040, hole_309x), hole_30fb),
            );

            let is_cjk_letter = v128_or(v128_or(han, hangul), kana);
            out = v128_bitselect(u8x16_splat(CJK_TAG), out, is_cjk_letter);
            resolved = v128_or(resolved, is_cjk_letter);
        }

        // 3-byte non-CJK: exact peel of the distinct (lead, b1-pair) blocks still present
        let is_lead3 = v128_andnot(in_range(b0, 0xE0, 0xEF), resolved);
        if v128_any_true(is_lead3) {
            let block_index = v128_or(
                i8x16_shl(v128_and(b1, u8x16_splat(1)), 6),
                v128_and(b2, u8x16_splat(0x3F)),
            );
            let pair = u8x16_shr(b1, 1); // b1>>1 — the 128-cp block-pair id
            let mut tags3 = u8x16_splat(MB);
            let mut unresolved = is_lead3;
            while v128_any_true(unresolved) {
                let lead = hmin(v128_bitselect(b0, ones, unresolved)); // smallest unresolved lead
                let lead_lanes = v128_and(unresolved, eq(b0, lead));
                let min_pair = hmin(v128_bitselect(pair, ones, lead_lanes)); // smallest pair within it
                let block_lanes = v128_and(lead_lanes, u8x16_eq(pair, u8x16_splat(min_pair)));
                let block = (lead - 0xE0) as usize * 32 + (min_pair & 0x1F) as usize;
                let uniform_tag = ATOM_TABLES.fast3_uni[block];
                let block_tags = if uniform_tag != 0xFF {
                    u8x16_splat(uniform_tag) // whole block is one tag
                } else {
                    let (lo, hi) = &ATOM_TABLES.fast3_mixed[ATOM_TABLES.fast3_slot[block] as usize];
                    lookup128(lo, hi, block_index)
                };
                tags3 = v128_bitselect(block_tags, tags3, block_lanes);
                unresolved = v128_andnot(unresolved, block_lanes); // drop the lanes just resolved
            }
            out = v128_bitselect(tags3, out, is_lead3);
            resolved = v128_or(resolved, v128_andnot(is_lead3, eq(tags3, MB)));
        }

        // residual multibyte lead → MB ; continuation byte → CONT
        let stray_lead = v128_andnot(ge(b0, 0xC0), resolved);
        out = v128_bitselect(u8x16_splat(MB), out, stray_lead);
        let is_cont = u8x16_eq(v128_and(b0, u8x16_splat(0xC0)), u8x16_splat(0x80));
        out = v128_bitselect(u8x16_splat(CONT), out, is_cont);

        if v128_any_true(eq(out, MB)) {
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
        tags[i] = ATOM_TABLES.classify_char(text, i);
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
        let mut pos = 0;
        while pos < n {
            if tags[pos] == MB {
                let cp = decode(text, pos);
                tags[pos] = if cp < 0x10000 {
                    ATOM_TABLES.bmp_tag(cp as u16)
                } else {
                    ATOM_TABLES.classify_char(text, pos)
                };
            }
            pos += 1;
        }
    }
}
