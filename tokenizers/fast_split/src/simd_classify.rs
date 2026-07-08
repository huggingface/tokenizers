use super::classify::TagScheme;

/// Hot path at each byte-width we compute min/max values which gives us an index to one of the table; the fallback
/// loops the SAME tables over the min..max range. This happens when we have  mixed scripts.
pub struct Tables {
    // ── ASCII (v < 0x80): direct 128-entry byte→tag, split 0..63 / 64..127.
    // This uses a trick to lookup more than 64 entries. we shift the bytes
    // we look for in the second lookup. If the index is > 64 the first lookup will return 0
    // but the second will then not be out of bound. Similarly if the first is in bound, the second
    // lookup will be out of bound.
    //
    // - byte v = 5:  tbl64(lo, 5) = lo[5];  tbl64(hi, 5-64=197) → 197≥64 → 0.  OR gives lo[5]. ✓
    // - byte v = 65: tbl64(lo, 65) → 65≥64 → 0; tbl64(hi, 65-64=1) = hi[1]. OR gives hi[1]. ✓
    pub ascii_lo: [u8; 64],
    pub ascii_hi: [u8; 64],

    // ── 2-byte (C2..DF): 8 groups of 4 leads (there are only 32 different lead bytes in a 2 byte long utf8);
    // Here again, in order to lookup into not 128 but 256 values we need 4 tables.
    // We decompose a 2 byte char as 110xxxyy 10yyyyyy. xxx gives the group (out of 8) while (yy)
    // gives the sub group (out of 4). Finally, yyyyyy gives the index (0..64) in the [u8; 64] that
    // gives the tag of the full 2-byte.
    pub group_tables: [[[u8; 64]; 4]; 8],

    // There are uniform and non uniform ranges in the 3-byte family.
    // - Georgian (E1 82/83), Cherokee, Coptic, Runic/Ogham, Hangul Jamo, Ethiopic (E1 88–9B, syllabary) — alphabets/syllabaries whose marks (if any) sit in separate blocks.
    // - CJK Han (E4–E9) and Hangul syllables (EB–EC) are all-Letter uniform too — but they go through the range path, not fast3.
    // - Pure symbol / unassigned regions → uniform SymOther.
    // 425 of 512 blocks are uniform (~83%); only 87 are mixed.
    // The rest need mixed slot and a similar trick than what we used above.
    pub fast3_uni: Box<[u8; 512]>, // per block: the tag if uniform, or 0xFF ("mixed, look deeper")
    pub fast3_slot: Box<[u16; 512]>, // per block: WHERE this block's 128-table lives in fast3_mixed (only if mixed)
    pub fast3_mixed: Vec<([u8; 64], [u8; 64])>, // the sparse list of 128-tables — one per MIXED block (~87), lo/hi halves

    // ── Cold fallback (astral 4-byte, CJK-letter holes): run-length-encoded BMP tag table,
    //    (run_start_cp, tag), binary-searched. ~1-3 KB vs a dense 64 KB LUT.
    pub bmp_rle: Vec<(u16, u8)>,
}

impl Tables {
    /// Build every table from the scheme's scalar `classify_char`. `tag(bytes)` = tag of the char that
    /// those bytes encode; tables are dense over the byte space, so invalid sequences just never occur.
    /// These table can be built for unicode script and atoms.
    pub fn build<S: TagScheme>() -> Tables {
        let tag = |bytes: &[u8]| S::classify_char(bytes, 0);

        // ASCII 128-entry table (two halves for the subtract-trick lookup)
        let (mut ascii_lo, mut ascii_hi) = ([0u8; 64], [0u8; 64]);
        for b in 0..64u8 {
            ascii_lo[b as usize] = tag(&[b]);
            ascii_hi[b as usize] = tag(&[64 + b]);
        }

        // 2-byte: 8 groups × 4 leads × 64 conts
        let mut group_tables = [[[0u8; 64]; 4]; 8];
        for g in 0..8usize {
            // 0xC0 is 110..... the utf8 header
            // g is the group: 110xxx.. *4 shifts it by 2
            let base = 0xC0u8 + (g as u8) * 4;
            for k in 0..4usize {
                for c in 0..64u8 {
                    // 0x80 is 10yyyyyy, its constructing the 2nd byte.
                    // We add `k` as its the sub group (0,1,2,3)
                    group_tables[g][k][c as usize] = tag(&[base + k as u8, 0x80 | c]);
                }
            }
        }
        // Finally we build the hardest of them all:
        let mut fast3_uni = Box::new([0u8; 512]);
        let mut fast3_slot = Box::new([0u16; 512]);
        let mut fast3_mixed: Vec<([u8; 64], [u8; 64])> = Vec::new();
        for lead in 0xE0u8..=0xEF {
            for p in 0..32u8 {
                let (b2e, b2o) = (0x80 + 2 * p, 0x80 + 2 * p + 1);
                let (mut lo, mut hi) = ([0u8; 64], [0u8; 64]);
                for c in 0..64u8 {
                    lo[c as usize] = tag(&[lead, b2e, 0x80 | c]);
                    hi[c as usize] = tag(&[lead, b2o, 0x80 | c]);
                }
                let idx = (lead - 0xE0) as usize * 32 + p as usize;
                if lo.iter().chain(hi.iter()).all(|&x| x == lo[0]) {
                    fast3_uni[idx] = lo[0];
                } else {
                    fast3_uni[idx] = 0xFF;
                    fast3_slot[idx] = fast3_mixed.len() as u16;
                    fast3_mixed.push((lo, hi));
                }
            }
        }

        // Cold fallback: dense BMP tag → run-length encode (emit only on tag change)
        let mut buf = [0u8; 4];
        let mut dense = vec![0u8; 0x10000];
        for cp in 0..0x10000u32 {
            if (0xD800..=0xDFFF).contains(&cp) {
                continue;
            }
            if let Some(c) = char::from_u32(cp) {
                dense[cp as usize] = tag(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        let mut bmp_rle: Vec<(u16, u8)> = Vec::new();
        for cp in 0..0x10000u32 {
            let a = dense[cp as usize];
            if bmp_rle.last().map_or(true, |&(_, la)| la != a) {
                bmp_rle.push((cp as u16, a));
            }
        }

        Tables {
            ascii_lo,
            ascii_hi,
            group_tables,
            fast3_uni,
            fast3_slot,
            fast3_mixed,
            bmp_rle,
        }
    }

    /// Cold-path BMP fallback: last run whose start ≤ cp (`bmp_rle[0].0 == 0`, so the index ≥ 1).
    #[inline]
    pub fn bmp_tag(&self, cp: u16) -> u8 {
        self.bmp_rle[self.bmp_rle.partition_point(|&(s, _)| s <= cp) - 1].1
    }
}

// ================================================================================================
// The smallest load we can do in neon is vld1q_u8, which handles 16bytes.
// We define primitives to be able to index into 4 x 16 byte vectors.
// The vqtbl4q_u8 allows for a table lookup
// ================================================================================================
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn tbl64(
    t: &[u8; 64],
    idx: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    // vld1q_u8_x4 = LD1 {v0.16b-v3.16b}: 64 CONTIGUOUS bytes into 4 lanes — the layout vqtbl4 wants.
    // (NOT vld4q_u8: that de-interleaves stride-4 into planes, which would scramble the table.)
    vqtbl4q_u8(vld1q_u8_x4(t.as_ptr()), idx)
}
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn tbl256(
    t: &[[u8; 64]; 4],
    idx: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vorrq_u8(
        vorrq_u8(
            tbl64(&t[0], idx),
            tbl64(&t[1], vsubq_u8(idx, vdupq_n_u8(64))),
        ),
        vorrq_u8(
            tbl64(&t[2], vsubq_u8(idx, vdupq_n_u8(128))),
            tbl64(&t[3], vsubq_u8(idx, vdupq_n_u8(192))),
        ),
    )
}
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn ascii_tbl(
    v: core::arch::aarch64::uint8x16_t,
    lo: &[u8; 64],
    hi: &[u8; 64],
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    vorrq_u8(tbl64(lo, v), tbl64(hi, vsubq_u8(v, vdupq_n_u8(64))))
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

#[inline]
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

/// Whole-buffer NEON classify, generic over the scheme. Tables come from `S::tables()`; the scalar
/// `S::classify_char` covers only the <32-byte tail and astral 4-byte codepoints. Byte-exact vs the
/// scalar walk. `classify::<S>` dispatches here on aarch64 (NEON is baseline there).
///
/// Per lane, `b0`/`b1`/`b2` are the 1st/2nd/3rd bytes of the (potential) UTF-8 char starting there —
/// i.e. the byte at the lane and the two after it (`vext` shifts the next chunk in).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn, non_snake_case)]
pub unsafe fn classify_neon<S: super::classify::TagScheme>(text: &[u8], tags: &mut [u8]) {
    use super::classify::char_len;
    use core::arch::aarch64::*;
    let (MB, CONT) = (S::MB, S::CONT);
    let tables = S::tables();

    let n = text.len();
    let mut mb_seen = false;
    let mut i = 0;
    while i + 32 <= n {
        let b0 = vld1q_u8(text.as_ptr().add(i));

        // ── ASCII fast path: whole chunk < 0x80 → one table, skip everything else ──
        if vmaxvq_u8(b0) < 0x80 {
            vst1q_u8(
                tags.as_mut_ptr().add(i),
                ascii_tbl(b0, &tables.ascii_lo, &tables.ascii_hi),
            );
            i += 16;
            continue;
        }

        let next = vld1q_u8(text.as_ptr().add(i + 16));
        let b1 = vextq_u8::<1>(b0, next); // byte at lane+1
        let b2 = vextq_u8::<2>(b0, next); // byte at lane+2
        let mut out = ascii_tbl(b0, &tables.ascii_lo, &tables.ascii_hi); // base (ASCII lanes correct; MB overwritten)
        let mut resolved = vdupq_n_u8(0); // lanes a multibyte handler has claimed

        // ── 2-byte (C2..DF): loop the lead-GROUP range; single group ⇒ 1 iteration ──
        let is_lead2 = eq(vandq_u8(b0, vdupq_n_u8(0xE0)), 0xC0);
        if any(is_lead2) {
            let group_index = vorrq_u8(
                vshlq_n_u8::<6>(vandq_u8(b0, vdupq_n_u8(3))),
                vandq_u8(b1, vdupq_n_u8(0x3F)),
            );
            let min_lead = vminvq_u8(vbslq_u8(is_lead2, b0, vdupq_n_u8(0xFF)));
            let max_lead = vmaxvq_u8(vbslq_u8(is_lead2, b0, vdupq_n_u8(0x00)));
            let mut tags2 = vdupq_n_u8(MB);
            let mut group = min_lead >> 2;
            while group <= (max_lead >> 2) {
                let this_group = vandq_u8(is_lead2, eq(vshrq_n_u8::<2>(b0), group));
                if any(this_group) {
                    let group_table = &tables.group_tables[(group & 7) as usize];
                    tags2 = vbslq_u8(this_group, tbl256(group_table, group_index), tags2);
                }
                group += 1;
            }
            out = vbslq_u8(is_lead2, tags2, out);
            resolved = is_lead2;
        }

        // ── CJK fast path (leads E3..ED = U+3000..U+DFFF) — ONLY when the scheme collapses all of CJK
        //    to one tag (Atoms → Letter). Schemes where CJK spans several tags (Scripts: Han/Hangul/
        //    Kana) set CJK_RANGE_TAG=None and skip this → the 3-byte tables below resolve E3..ED.
        //    OPTIMISTIC bulk: flags only DEFINITELY-CJK lanes and leaves boundary/hole codepoints
        //    unresolved for the exact 3-byte tables — never over-claims, so the result stays byte-exact.
        if let Some(cjk_tag) = S::CJK_RANGE_TAG {
            let in_cjk_leads = in_range(b0, 0xE3, 0xED);
            if any(in_cjk_leads) {
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
                out = vbslq_u8(is_cjk_letter, vdupq_n_u8(cjk_tag), out);
                resolved = vorrq_u8(resolved, is_cjk_letter);
            }
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
                        tbl64(lo, block_index),
                        tbl64(hi, vsubq_u8(block_index, vdupq_n_u8(64))),
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

        mb_seen |= any(eq(out, MB));
        vst1q_u8(tags.as_mut_ptr().add(i), out);
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
        tags[i] = S::classify_char(text, i);
        let w = char_len(b);
        for j in 1..w {
            if i + j < n {
                tags[i + j] = CONT;
            }
        }
        i += w;
    }

    // ── MB fixup: resolve every lane the SIMD left as MB (CJK holes, astral) ──
    if mb_seen {
        let mut pos = 0;
        while pos < n {
            if tags[pos] == MB {
                let cp = decode(text, pos);
                tags[pos] = if cp < 0x10000 {
                    tables.bmp_tag(cp as u16)
                } else {
                    S::classify_char(text, pos)
                };
            }
            pos += 1;
        }
    }
}
