use crate::tables::Tables;

// ================================================================================================
// The smallest load we can do in neon is vld1q_u8, which handles 16bytes.
// We define primitives to be able to index into 4 x 16 byte vectors.
// The vqtbl4q_u8 allows for that exactly.
// ================================================================================================
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
/// This is a handle to simulate a 256 lookup using 4 x 64 lookups.
unsafe fn tbl256(
    t: &[[u8; 64]; 4],
    idx: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vorrq_u8(
        vorrq_u8(
            vqtbl4q_u8(vld1q_u8_x4(t[0].as_ptr()), idx),
            vqtbl4q_u8(vld1q_u8_x4(t[1].as_ptr()), vsubq_u8(idx, vdupq_n_u8(64))),
        ),
        vorrq_u8(
            vqtbl4q_u8(vld1q_u8_x4(t[2].as_ptr()), vsubq_u8(idx, vdupq_n_u8(128))),
            vqtbl4q_u8(vld1q_u8_x4(t[3].as_ptr()), vsubq_u8(idx, vdupq_n_u8(192))),
        ),
    )
}
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
/// This is using the substraction trick to retrive the tag of any ascii chars.
/// We look in the first low table, any u8 value that is <64 will give 0. So in the first table
/// only the bytes whose value is < 64 will give a non 0 tag. In the second lookup we shift all
/// byte value by 64 so that all bytes>64 can now properly index the 64 entry high table.
unsafe fn ascii_tbl(
    v: core::arch::aarch64::uint8x16_t,
    lo: &[u8; 64],
    hi: &[u8; 64],
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vorrq_u8(
        vqtbl4q_u8(vld1q_u8_x4(lo.as_ptr()), v),
        vqtbl4q_u8(vld1q_u8_x4(hi.as_ptr()), vsubq_u8(v, vdupq_n_u8(64))),
    )
}

// ── lane predicates (each returns an all-ones / all-zeros mask per lane) — the readable range compares ──
/// Lanes equal to `byte`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn eq(bytes: core::arch::aarch64::uint8x16_t, byte: u8) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vceqq_u8(bytes, vdupq_n_u8(byte))
}
/// Lanes `>= lo` (unsigned).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn ge(bytes: core::arch::aarch64::uint8x16_t, lo: u8) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vcgeq_u8(bytes, vdupq_n_u8(lo))
}
/// Lanes `<= hi` (unsigned).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn le(bytes: core::arch::aarch64::uint8x16_t, hi: u8) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vcleq_u8(bytes, vdupq_n_u8(hi))
}
/// Lanes with `lo <= byte <= hi` (unsigned, inclusive).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn in_range(
    bytes: core::arch::aarch64::uint8x16_t,
    lo: u8,
    hi: u8,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vandq_u8(ge(bytes, lo), le(bytes, hi))
}
/// `true` iff any lane is set (mask is all-ones / all-zeros per lane).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn any(mask: core::arch::aarch64::uint8x16_t) -> bool {
    use core::arch::aarch64::*;
    vmaxvq_u8(mask) != 0
}

#[cfg(target_arch = "aarch64")] // only `classify_neon`'s MB-fixup calls this; dead on other arches
#[inline]
/// This is just for any char lenght, decode a utf8 to its actual value.
/// TLDR removing the utf8 headers to get the unicode.
fn decode(t: &[u8], i: usize) -> u32 {
    let b = t[i] as u32;
    match super::classify::char_len(t[i]) {
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

/// This is the key classification function. We built the tables, and here we leverage them.
/// The core idea is to assume 16 bytes are gonna be on average all from the same script. This
/// means that the same classification rule can be applied for all of them. Otherwise, we run all
/// the classification rules.
///
/// Rules are associated with the total number of bytes of the utf8 character:
///     - 1 bytes: all the ascii table holds in a table of 128 tags.
///     - 2 bytes: there are 2048 2-byte chars. We can do a fast lookup in such a big table, so we
///     extract which table needs to be looked at from the leading bytes. 110xxxyy 10zzzzzz: xxx is the
///     group, yyzzzzzz the index. We have to loop on the different xxx because
///     in SIMD we can't do any indexing in tables with an index > 256. But again, characters of
///     the same scripts will most often have the same leading xxx:
// ______________________________________________________________________
// | xxx | U+ range  | scripts                                            |
// |_____|___________|____________________________________________________|
// | 000 | 0080-00FF | Latin-1 Supplement                                 |
// | 001 | 0100-01FF | Latin Extended-A, Latin Extended-B (start)         |
// | 010 | 0200-02FF | Latin Extended-B (end), IPA Ext, Spacing Modifiers |
// | 011 | 0300-03FF | Combining Diacritical Marks, Greek & Coptic        |
// | 100 | 0400-04FF | Cyrillic                                           |
// | 101 | 0500-05FF | Cyrillic Supplement, Armenian, Hebrew              |
// | 110 | 0600-06FF | Arabic                                             |
// | 111 | 0700-07FF | Syriac, Arabic Supplement, Thaana, NKo             |
// |_____|___________|____________________________________________________|
///     We loop until we looked up the tables of each xxx values present in the 16 bytes.
///     - 3 bytes: there are 3 cases, either the range is known to be cjk, either the range of the leading bytes allow direct
///     classification (meanin from just the x's in 1110xxx 10xxxxxx 10yyyyyy we are able to
///     determine the tag/class) or we need to do a lookup. Again, `xxxxxxxxx` gives us the index
///     in `fast3_mixed`, and yyyyyy which element to lookup inside that table.
//  ________________________________________________
// | wwww | lead | U+ range  | kind                 |
// |______|______|___________|______________________|
// | 0101 | E5   | 5000-5FFF | Han ideographs       |
// | 0110 | E6   | 6000-6FFF | Han ideographs       |
// | 0111 | E7   | 7000-7FFF | Han ideographs       |
// | 1000 | E8   | 8000-8FFF | Han ideographs       |
// | 1001 | E9   | 9000-9FFF | Han ideographs       |
// | 1011 | EB   | B000-BFFF | Hangul syllables     |
// | 1100 | EC   | C000-CFFF | Hangul syllables     |
// |______|______|___________|______________________|
/// For the above ranges, we only need the first bytes.
/// The reason we don't use the same table format for 2 or 3 byte?
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn, non_snake_case)]
pub unsafe fn classify_neon<const CONT: u8, const MB: u8, const CJK_TAG: u8>(
    text: &[u8],
    tags: &mut [u8],
    tables: &Tables,
) {
    use super::classify::char_len;
    use core::arch::aarch64::*;
    let n = text.len();
    let mut i = 0;
    while i + 32 <= n {
        // 1. load the first byte in each lane. 16 lanes, each now own 1 byte. All operations are
        //    run in parallel.
        let b0 = vld1q_u8(text.as_ptr().add(i));

        // ASCII fast path: whole chunk < 0x80 → one table, skip everything else
        if vmaxvq_u8(b0) < 0x80 {
            // vst1q_u8 stores the value from ascii_tbl into tags.
            vst1q_u8(
                tags.as_mut_ptr().add(i),
                ascii_tbl(b0, &tables.ascii_lo, &tables.ascii_hi),
            );
            i += 16;
            continue;
        }

        // Not all ascii, so let's default the tags by computing ASCII.
        let mut out = ascii_tbl(b0, &tables.ascii_lo, &tables.ascii_hi); // base (ASCII lanes correct; MB overwritten)

        // Let's load the next chunk of 16 bytes
        let next = vld1q_u8(text.as_ptr().add(i + 16));
        // vext::<N>  does: cat(bo[N..], next[..N])
        let b1 = vextq_u8::<1>(b0, next); // byte at lane+1
        let b2 = vextq_u8::<2>(b0, next); // byte at lane+2
        // each lane (b0[i]), a lane is `i`, can now access the next byte's value.

        // We'll construct the resolved mask
        let mut resolved = vdupq_n_u8(0); // lanes a multibyte handler has claimed

        // ── 2-byte (C2..DF): here we could have differnt lead groups per byte.
        // Since we are in SIMD, we need to potentially iterate over the group index
        // represented in all lanes. So until there are no groups left, we compute the
        // min group, run the lookup, continue. That's because we are doing lookups in
        // 2 level table, which SIMD does not support (each of the 16 lanes might need to peekd
        //   into a different table, which is not possible).
        let is_lead2 = eq(vandq_u8(b0, vdupq_n_u8(0xE0)), 0xC0);
        if any(is_lead2) {
            // this is extracting yyyyyyyy out of 110xxxyy 10yyyyyy.
            // 110xxxyy & 000000111 << 6 -> yy000000
            // 10yyyyyy & 001111111      -> 00yyyyyy
            //                            | yyyyyyyy -> used to index the 256 lookup table
            let group_index = vorrq_u8(
                vshlq_n_u8::<6>(vandq_u8(b0, vdupq_n_u8(3))),
                vandq_u8(b1, vdupq_n_u8(0x3F)),
            );
            let grp = vshrq_n_u8::<2>(b0); // lead-group id per lane (the xxx)
            // tag2 are the tags for the 2-bytes long chars of this group
            let mut tags2 = vdupq_n_u8(MB);
            // which of these are still unresolved
            let mut unresolved = is_lead2;
            while any(unresolved) {
                // we can't lookup per lane more than 256 entries, so we go through the xxx groups
                // of all the present bytes. this is worst case 8 different groups in aversarial,
                // best case a single lookup.
                let group = vminvq_u8(vbslq_u8(unresolved, grp, vdupq_n_u8(0xFF))); // min group present
                let group_lanes = vandq_u8(unresolved, eq(grp, group)); // lanes of exactly this group
                let group_table = &tables.group_tables[(group & 7) as usize];
                tags2 = vbslq_u8(group_lanes, tbl256(group_table, group_index), tags2);
                unresolved = vbicq_u8(unresolved, group_lanes); // drop the lanes just resolved
            }
            out = vbslq_u8(is_lead2, tags2, out);
            resolved = is_lead2;
        }

        if CJK_TAG != crate::classify::NO_CJK && any(in_range(b0, 0xE3, 0xED)) {
            // Han — U+4000..U+9FFF (CJK Unified Ideographs + the Ext-A tail), minus the one non-
            // ideograph hole U+4DC0..U+4DFF (Yijing Hexagram Symbols), which encodes as E4 B7 xx.
            let han = vbicq_u8(
                in_range(b0, 0xE4, 0xE9),
                vandq_u8(eq(b0, 0xE4), eq(b1, 0xB7)),
            );

            // Hangul Syllables (U+AC00..U+D7A3), split across leads EA..ED:
            //   EB..EC        → U+B000..U+CFFF  (whole middle — every lane a syllable)
            //   EA, b1 >= B0  → U+AC00..U+AFFF  (syllables begin at AC00; U+A000..ABFF below is excluded)
            //   ED, b1 <= 9D  → U+D000..U+D77F  (syllables; the U+D780.. tail is left to the exact tables)
            let hangul = vorrq_u8(
                vorrq_u8(
                    in_range(b0, 0xEB, 0xEC),
                    vandq_u8(eq(b0, 0xEA), ge(b1, 0xB0)),
                ),
                vandq_u8(eq(b0, 0xED), le(b1, 0x9D)),
            );

            // Kana — Hiragana + Katakana U+3040..U+30FF (lead E3, b1 in 81..83), minus the
            // non-letter holes inside that block:
            //   U+3040          reserved                 (E3 81 80)
            //   U+3097..U+309C  unassigned + combining    (E3 82 97..9C)
            //   U+30A0          double hyphen (Punct)      (E3 82 A0)
            //   U+30FB          middle dot (Punct)         (E3 83 BB)
            let hole_3040 = vandq_u8(eq(b1, 0x81), eq(b2, 0x80));
            let hole_309x = vandq_u8(
                eq(b1, 0x82),
                vorrq_u8(in_range(b2, 0x97, 0x9C), eq(b2, 0xA0)),
            );
            let hole_30fb = vandq_u8(eq(b1, 0x83), eq(b2, 0xBB));
            let kana = vbicq_u8(
                vandq_u8(eq(b0, 0xE3), in_range(b1, 0x81, 0x83)),
                vorrq_u8(vorrq_u8(hole_3040, hole_309x), hole_30fb),
            );

            let is_cjk_letter = vorrq_u8(vorrq_u8(han, hangul), kana);
            out = vbslq_u8(is_cjk_letter, vdupq_n_u8(CJK_TAG), out);
            resolved = vorrq_u8(resolved, is_cjk_letter);
        }

        // ── 3-byte non-CJK: peel the EXACT distinct (lead, b1-pair) blocks present. Pick min (lead,
        //    then min b1-pair) among unresolved lanes, resolve that whole 128-cp block, mask it out,
        //    repeat. Iterations = distinct-block count (≤5-6/chunk), independent of how far apart the
        //    blocks are → adversarial-proof, no box guard, and faster than the min..max range. ──
        let is_lead3 = vbicq_u8(in_range(b0, 0xE0, 0xEF), resolved);
        if any(is_lead3) {
            // within-block index: block_index = ((b1&1)<<6) | (b2&0x3F);  pair = b1>>1 identifies the b1-pair
            let block_index = vorrq_u8(
                vshlq_n_u8::<6>(vandq_u8(b1, vdupq_n_u8(1))),
                vandq_u8(b2, vdupq_n_u8(0x3F)),
            );
            let pair = vshrq_n_u8::<1>(b1);
            let mut tags3 = vdupq_n_u8(MB);
            let mut unresolved = is_lead3;
            while any(unresolved) {
                let lead = vminvq_u8(vbslq_u8(unresolved, b0, vdupq_n_u8(0xFF))); // min lead among unresolved
                let lead_lanes = vandq_u8(unresolved, eq(b0, lead));
                let min_pair = vminvq_u8(vbslq_u8(lead_lanes, pair, vdupq_n_u8(0xFF))); // min b1-pair within it
                let block_lanes = vandq_u8(lead_lanes, eq(pair, min_pair)); // lanes of exactly this block
                let block = (lead - 0xE0) as usize * 32 + (min_pair & 0x1F) as usize;
                let uniform_tag = tables.fast3_uni[block];
                let block_tags = if uniform_tag != 0xFF {
                    vdupq_n_u8(uniform_tag) // whole block is one tag
                } else {
                    let (lo, hi) = &tables.fast3_mixed[tables.fast3_slot[block] as usize];
                    vorrq_u8(
                        vqtbl4q_u8(vld1q_u8_x4(lo.as_ptr()), block_index),
                        vqtbl4q_u8(
                            vld1q_u8_x4(hi.as_ptr()),
                            vsubq_u8(block_index, vdupq_n_u8(64)),
                        ),
                    )
                };
                tags3 = vbslq_u8(block_lanes, block_tags, tags3);
                unresolved = vbicq_u8(unresolved, block_lanes); // drop the lanes just resolved
            }
            out = vbslq_u8(is_lead3, tags3, out);
            resolved = vorrq_u8(resolved, vbicq_u8(is_lead3, eq(tags3, MB)));
        }

        // ── residual multibyte lead → MB ; continuation byte → CONT ──
        let stray_lead = vbicq_u8(ge(b0, 0xC0), resolved);
        out = vbslq_u8(stray_lead, vdupq_n_u8(MB), out);
        let is_cont = eq(vandq_u8(b0, vdupq_n_u8(0xC0)), 0x80);
        out = vbslq_u8(is_cont, vdupq_n_u8(CONT), out);

        vst1q_u8(tags.as_mut_ptr().add(i), out);
        // Per-chunk MB fixup: astral (4-byte, the vector path only gathers b0/b1/b2) and any RLE-deferred
        // lead. Rare on real text → the branch is ~always-false and free; when taken, this chunk's bytes
        // are still hot in L1 (vs a second full pass over the whole buffer). Only the ≤16 MB lanes are fixed.
        if any(eq(out, MB)) {
            for j in 0..16 {
                if tags[i + j] == MB {
                    let cp = decode(text, i + j);
                    tags[i + j] = if cp < 0x10000 {
                        tables.bmp_tag(cp as u16)
                    } else {
                        tables.classify_char(text, i + j)
                    };
                }
            }
        }
        i += 16;
    }

    // ── scalar tail (< 32 bytes) ──
    while i < n {
        let b = text[i];
        if b & 0xC0 == 0x80 {
            tags[i] = CONT;
            i += 1;
            continue;
        }
        tags[i] = tables.classify_char(text, i);
        let w = char_len(b);
        for j in 1..w {
            if i + j < n {
                tags[i + j] = CONT;
            }
        }
        i += w;
    }
}
