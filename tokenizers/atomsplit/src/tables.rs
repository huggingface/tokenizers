/// Dense classify tables. A 16-byte SIMD chunk is tagged at each byte width (ASCII / 2-byte / 3-byte)
/// with as few table lookups as possible, exploiting that real text is near-homogeneous — almost every
/// lane in a chunk is in the SAME script block, so one lookup tags all of them.
///
/// Mixed chunks are handled by PEELING the blocks that are actually present, not by scanning a range:
///   1. `vminvq` the block-id over the still-unresolved lanes → the smallest block present in the chunk;
///   2. look up that ONE block's table and write its tag to exactly the lanes belonging to it;
///   3. clear those lanes (`vbic`) and repeat until no unresolved lanes remain.
///
/// So the number of lookups = the count of DISTINCT blocks in the chunk (usually 1 → that's the fast
/// path; at most a handful), independent of how far apart the scripts sit: Latin-1 + Cyrillic is 2 peels,
/// not "every group in between." No bounds guard needed — the earlier `min..=max` loop instead stepped
/// once per block in the span, wasting a step on every empty gap between the scripts actually present.
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
    pub fast3_uni: [u8; 512], // per block: the tag if uniform, or 0xFF ("mixed, look deeper")
    pub fast3_slot: [u16; 512], // per block: WHERE this block's 128-table lives in fast3_mixed (only if mixed)
    pub fast3_mixed: &'static [([u8; 64], [u8; 64])], // the sparse list of 128-tables — one per MIXED block (~87), lo/hi halves

    // ── Cold fallback (astral 4-byte, CJK-letter holes): run-length-encoded BMP tag table,
    //    (run_start_cp, tag), binary-searched. ~1-3 KB vs a dense 64 KB LUT.
    pub bmp_rle: &'static [(u16, u8)],

    // 4-byte astral (cp ≥ 0x10000): run-length-encoded (start_cp, atom), binary-searched.
    pub astral: &'static [(u32, u8)],
}

impl Tables {
    /// Scalar reader over the dense tables — the portable twin of the SIMD kernel, and the single
    /// source for `TagScheme::classify_char`, the <32-byte SIMD tail, and the SIMD MB-fixup. Returns
    /// the atom for the char starting at `text[i]`. Byte-exact with the SIMD path (same tables).
    #[inline]
    pub fn classify_char(&self, text: &[u8], i: usize) -> u8 {
        let b0 = text[i];
        if b0 < 0x80 {
            return if b0 < 64 {
                self.ascii_lo[b0 as usize]
            } else {
                self.ascii_hi[(b0 - 64) as usize]
            };
        }
        if b0 < 0xE0 {
            // 110xxxzz 10yyyyyy -> xxx is the group, zz is the sub group, yyyyyy the index into
            // the table (cont)
            // 2-byte: group_tables[(lead>>2)&7][lead&3][cont&0x3F]
            let cont = text[i + 1] & 0x3F;
            return self.group_tables[((b0 >> 2) & 7) as usize][(b0 & 3) as usize][cont as usize];
        }
        if b0 < 0xF0 {
            // 1110xxx 10xxxxxx 10yyyyyy-> xxxxxxxxx the 9 lead bits give the group, yyyyyy the sub
            // group. The 9 bit give 0..511 index. If its uniform, that will give a tag. If not,
            // the value stored in fast3_uni will be 0xFF and we need to index fast3_slot with the
            // index. `fast3_slot[index]` gives where to look in `fast3_mixed`'s first table, the
            // second one is indexed by 0..63 the continuation bytes. yyyyyy.
            // Example:
            // 3-byte: block = (lead-0xE0)*32 + ((b1>>1)&0x1F)
            // if fast3_uni[block] == 0xFF : tag = fast3_mixed[fast3_slot[block]][yyyyyy]
            let b1 = text[i + 1];
            let c = (text[i + 2] & 0x3F) as usize;
            let block = (b0 - 0xE0) as usize * 32 + ((b1 >> 1) & 0x1F) as usize;
            let uni = self.fast3_uni[block];
            if uni != 0xFF {
                return uni;
            }
            let (lo, hi) = &self.fast3_mixed[self.fast3_slot[block] as usize];
            return if b1 & 1 == 0 { lo[c] } else { hi[c] };
        }
        // 4-byte astral
        let cp = ((b0 as u32 & 0x07) << 18)
            | ((text[i + 1] as u32 & 0x3F) << 12)
            | ((text[i + 2] as u32 & 0x3F) << 6)
            | (text[i + 3] as u32 & 0x3F);
        self.astral[self.astral.partition_point(|&(s, _)| s <= cp) - 1].1
    }

    /// Cold-path BMP fallback: last run whose start ≤ cp (`bmp_rle[0].0 == 0`, so the index ≥ 1).
    #[inline]
    pub fn bmp_tag(&self, cp: u16) -> u8 {
        self.bmp_rle[self.bmp_rle.partition_point(|&(s, _)| s <= cp) - 1].1
    }
}
