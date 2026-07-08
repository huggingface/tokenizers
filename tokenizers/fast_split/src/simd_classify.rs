use super::classify::{Atom, TagScheme};

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
// Reference SIMD classify body (Atom scheme). Every width = upfront min/max dispatch → one table;
// the fallback loops the SAME tables over the min..max range (single-script chunk ⇒ 1 iteration).
// `tag_scalar` (the scalar `classify_char`) is used ONLY for the <32-byte tail and astral (4-byte).
// ================================================================================================
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn tbl64(
    t: &[u8; 64],
    idx: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    let p = t.as_ptr();
    let r = uint8x16x4_t(
        vld1q_u8(p),
        vld1q_u8(p.add(16)),
        vld1q_u8(p.add(32)),
        vld1q_u8(p.add(48)),
    );
    vqtbl4q_u8(r, idx)
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

/// Reference body for review — Atom scheme. `tag_scalar` = the scalar `classify_char` (friend's), used
/// only for the <32-byte tail and astral 4-byte codepoints. Wire to the trait via a `OnceLock<Tables>`.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn classify_neon_ref(
    text: &[u8],
    tags: &mut [u8],
    tb: &Tables,
    tag_scalar: impl Fn(&[u8], usize) -> u8,
) {
    use super::classify::char_len;
    use core::arch::aarch64::*;
    const CONT: u8 = Atom::Cont as u8;
    const MB: u8 = Atom::MultiByte as u8;
    const LETTER: u8 = Atom::Letter as u8;

    let n = text.len();
    let mut mb_seen = false;
    let mut i = 0;
    while i + 32 <= n {
        let v = vld1q_u8(text.as_ptr().add(i));

        // ── ASCII fast path: whole chunk < 0x80 → one table, skip everything else ──
        if vmaxvq_u8(v) < 0x80 {
            vst1q_u8(
                tags.as_mut_ptr().add(i),
                ascii_tbl(v, &tb.ascii_lo, &tb.ascii_hi),
            );
            i += 16;
            continue;
        }

        let vn = vld1q_u8(text.as_ptr().add(i + 16));
        let b2 = vextq_u8::<1>(v, vn); // each lane's next byte
        let b3 = vextq_u8::<2>(v, vn); // each lane's next-next byte
        let mut out = ascii_tbl(v, &tb.ascii_lo, &tb.ascii_hi); // base (ASCII lanes correct; MB overwritten)
        let mut res = vdupq_n_u8(0); // lanes a multibyte handler has claimed

        // ── 2-byte (C2..DF): loop the lead-GROUP range; single group ⇒ 1 iteration ──
        let is2 = vceqq_u8(vandq_u8(v, vdupq_n_u8(0xE0)), vdupq_n_u8(0xC0));
        if vmaxvq_u8(is2) != 0 {
            let idxg = vorrq_u8(
                vshlq_n_u8::<6>(vandq_u8(v, vdupq_n_u8(3))),
                vandq_u8(b2, vdupq_n_u8(0x3F)),
            );
            let minl = vminvq_u8(vbslq_u8(is2, v, vdupq_n_u8(0xFF)));
            let maxl = vmaxvq_u8(vbslq_u8(is2, v, vdupq_n_u8(0x00)));
            let mut c2 = vdupq_n_u8(MB);
            let mut g = minl >> 2;
            while g <= (maxl >> 2) {
                let gg = vandq_u8(is2, vceqq_u8(vshrq_n_u8::<2>(v), vdupq_n_u8(g)));
                if vmaxvq_u8(gg) != 0 {
                    c2 = vbslq_u8(gg, tbl256(&tb.group_tables[(g & 7) as usize], idxg), c2);
                }
                g += 1;
            }
            out = vbslq_u8(is2, c2, out);
            res = is2;
        }

        // ── CJK letters (E3..ED) via lead-byte ranges (one range beats looping 6 Han leads) ──
        let iscjk = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xE3)), vcleq_u8(v, vdupq_n_u8(0xED)));
        if vmaxvq_u8(iscjk) != 0 {
            let han = vbicq_u8(
                vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xE4)), vcleq_u8(v, vdupq_n_u8(0xE9))),
                vandq_u8(
                    vceqq_u8(v, vdupq_n_u8(0xE4)),
                    vceqq_u8(b2, vdupq_n_u8(0xB7)),
                ),
            );
            let hg = vorrq_u8(
                vorrq_u8(
                    vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xEB)), vcleq_u8(v, vdupq_n_u8(0xEC))),
                    vandq_u8(
                        vceqq_u8(v, vdupq_n_u8(0xEA)),
                        vcgeq_u8(b2, vdupq_n_u8(0xB0)),
                    ),
                ),
                vandq_u8(
                    vceqq_u8(v, vdupq_n_u8(0xED)),
                    vcleq_u8(b2, vdupq_n_u8(0x9D)),
                ),
            );
            let e1 = vandq_u8(
                vceqq_u8(b2, vdupq_n_u8(0x81)),
                vceqq_u8(b3, vdupq_n_u8(0x80)),
            );
            let e2 = vandq_u8(
                vceqq_u8(b2, vdupq_n_u8(0x82)),
                vorrq_u8(
                    vandq_u8(
                        vcgeq_u8(b3, vdupq_n_u8(0x97)),
                        vcleq_u8(b3, vdupq_n_u8(0x9C)),
                    ),
                    vceqq_u8(b3, vdupq_n_u8(0xA0)),
                ),
            );
            let e3 = vandq_u8(
                vceqq_u8(b2, vdupq_n_u8(0x83)),
                vceqq_u8(b3, vdupq_n_u8(0xBB)),
            );
            let kana = vbicq_u8(
                vandq_u8(
                    vceqq_u8(v, vdupq_n_u8(0xE3)),
                    vandq_u8(
                        vcgeq_u8(b2, vdupq_n_u8(0x81)),
                        vcleq_u8(b2, vdupq_n_u8(0x83)),
                    ),
                ),
                vorrq_u8(vorrq_u8(e1, e2), e3),
            );
            let cjkl = vorrq_u8(vorrq_u8(han, hg), kana);
            out = vbslq_u8(cjkl, vdupq_n_u8(LETTER), out);
            res = vorrq_u8(res, cjkl);
        }

        // ── 3-byte non-CJK: loop the (lead × b2-pair) box; single script ⇒ 1×1 (fast path). ──
        // GUARD: an adversarial multi-script chunk could make the box up to 16×32 — cap it, and let
        // over-wide chunks fall through to MB → scalar fixup instead of spinning the nested loop.
        let is3 = vandq_u8(
            vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xE0)), vcleq_u8(v, vdupq_n_u8(0xEF))),
            vmvnq_u8(res),
        );
        if vmaxvq_u8(is3) != 0 {
            let minld = vminvq_u8(vbslq_u8(is3, v, vdupq_n_u8(0xFF)));
            let maxld = vmaxvq_u8(vbslq_u8(is3, v, vdupq_n_u8(0x00)));
            let minb2 = vminvq_u8(vbslq_u8(is3, b2, vdupq_n_u8(0xFF)));
            let maxb2 = vmaxvq_u8(vbslq_u8(is3, b2, vdupq_n_u8(0x00)));
            let box_area =
                (maxld - minld + 1) as usize * ((maxb2 >> 1) - (minb2 >> 1) + 1) as usize;
            if box_area <= 8 {
                let sel = vorrq_u8(
                    vshlq_n_u8::<6>(vandq_u8(b2, vdupq_n_u8(1))),
                    vandq_u8(b3, vdupq_n_u8(0x3F)),
                );
                let vp = vshrq_n_u8::<1>(b2);
                let mut c3 = vdupq_n_u8(MB);
                let mut lead = minld;
                while lead <= maxld {
                    let gl = vandq_u8(is3, vceqq_u8(v, vdupq_n_u8(lead)));
                    if vmaxvq_u8(gl) != 0 {
                        let mut pr = minb2 >> 1;
                        while pr <= (maxb2 >> 1) {
                            let gp = vandq_u8(gl, vceqq_u8(vp, vdupq_n_u8(pr)));
                            if vmaxvq_u8(gp) != 0 {
                                let idx = (lead - 0xE0) as usize * 32 + (pr & 0x1F) as usize;
                                let k = tb.fast3_uni[idx];
                                let cl = if k != 0xFF {
                                    vdupq_n_u8(k) // uniform (lead,b2-pair): one constant
                                } else {
                                    let (lo, hi) = &tb.fast3_mixed[tb.fast3_slot[idx] as usize];
                                    vorrq_u8(
                                        tbl64(lo, sel),
                                        tbl64(hi, vsubq_u8(sel, vdupq_n_u8(64))),
                                    )
                                };
                                c3 = vbslq_u8(gp, cl, c3);
                            }
                            pr += 1;
                        }
                    }
                    lead += 1;
                }
                out = vbslq_u8(is3, c3, out);
                res = vorrq_u8(res, vandq_u8(is3, vmvnq_u8(vceqq_u8(c3, vdupq_n_u8(MB)))));
            }
            // else: box too wide → is3 lanes stay unclaimed → become MB below → scalar fixup.
        }

        // ── residual multibyte lead → MB ; continuation byte → CONT ──
        let leadmb = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xC0)), vmvnq_u8(res));
        out = vbslq_u8(leadmb, vdupq_n_u8(MB), out);
        let cont = vceqq_u8(vandq_u8(v, vdupq_n_u8(0xC0)), vdupq_n_u8(0x80));
        out = vbslq_u8(cont, vdupq_n_u8(CONT), out);

        mb_seen |= vmaxvq_u8(vceqq_u8(out, vdupq_n_u8(MB))) != 0;
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
        tags[i] = tag_scalar(text, i);
        let w = char_len(b);
        for j in 1..w {
            if i + j < n {
                tags[i + j] = CONT;
            }
        }
        i += w;
    }

    // ── MB fixup: resolve every lane the SIMD left as MB (CJK holes, guard bail-outs, astral) ──
    if mb_seen {
        let mut k = 0;
        while k < n {
            if tags[k] == MB {
                let cp = decode(text, k);
                tags[k] = if cp < 0x10000 {
                    tb.bmp_tag(cp as u16)
                } else {
                    tag_scalar(text, k)
                };
            }
            k += 1;
        }
    }
}

/// Trait entry point for the Atom scheme (`Atoms::classify_neon` forwards here). Builds the Atom
/// `Tables` once (lazily), then runs the reference body with the scalar `classify_char` supplying the
/// <32-byte tail and astral codepoints.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn classify_neon(text: &[u8], tags: &mut [u8]) {
    use super::classify::{Atoms, TagScheme};
    use std::sync::OnceLock;
    static TABLES: OnceLock<Tables> = OnceLock::new();
    let tb = TABLES.get_or_init(Tables::build::<Atoms>);
    classify_neon_ref(text, tags, tb, <Atoms as TagScheme>::classify_char);
}

// out = vbsl(v==0x09 | v==0x0B | v==0x0C,  WsOther,    out)
// out = vbsl(v==0x20,                      Space,      out)
// out = vbsl(digit(0x30..0x39),            NumWord,    out)
// out = vbsl((v|0x20) in 0x61..0x7A,       Letter,     out)   // case-fold trick for a-z
// out = vbsl(v==0x5F,                      Connector,  out)   // '_'
// out = vbsl(v==0x27,                      Apostrophe, out)   // '\''
// 4 different SIMD function to implement. Each take text and tags and produce a tag value per char
//   in the tect right?
// Continuation-byte sentinel is written to every non-lead byte.
// There are 15 different ATOMS. which are combined to create the different rules.
// The tags is the buffer where we'll right the atom stream.
// Let's start with the ascii nibble-shuffle
// It is the first that will run on all lanes.
// out = ascii_classify(v)                 // all 16 lanes get an ASCII atom (wrong for non-ASCII lanes)
// out = vbsl(is_2byte_mask, twobyte(v), out)   // overwrite ONLY the 2-byte-lead lanes
// out = vbsl(is_cjk_mask,   cjk(v),     out)   // overwrite ONLY the CJK lanes
// out = vbsl(is_cont_mask,  CONT,       out)   // overwrite continuation bytes
// if vmaxvq_u8(is_2byte_mask) != 0 { ...the 2-byte work + blend... }
// this is scalar but locality of script means it predicts fairly well.
// the only ambiguous locations are script boundaries: from english to say chinese and etc, but
// they happen for a single data stream.
// The is_2byte mask — yes, cheap, just not literally one op
//
// A 2-byte lead is 110xxxxx (0xC2–0xDF), so the test is (v & 0xE0) == 0xC0:
// is2 = vceqq_u8(vandq_u8(v, vdupq_n_u8(0xE0)), vdupq_n_u8(0xC0))   // vand + vceq = 2 ops
// → a per-lane mask (0xFF where it's a 2-byte lead, 0x00 elsewhere). Your intuition holds: a couple of elementwise ops → a mask. Same shape for the others:
// - CJK lead range: vcgeq(v,0xE3) & vcleq(v,0xED) (range = 2 compares + 1 and).
// - continuation byte: (v & 0xC0) == 0x80.
//
// All cheap, all produce per-lane masks you feed to vbsl.
//
// "Lane-wise" for 2/3-byte — yes, but you need vext first
//
// Here's the subtlety. The classification math is lane-wise (elementwise: vand, vceq, vcge, vqtbl, vshl — each lane independent). But a multibyte character spans several lanes: the lead is in lane i, its continuation byte(s) in lanes i+1 (and i+2 for 3-byte). To classify the char at the lead lane, that lane needs the continuation bytes — which live in adjacent lanes.
//
// That cross-lane move is vext (a byte-shift of the whole vector):
// let b2 = vextq_u8(v, vn, 1);   // at each lane i, this holds byte[i+1]  (the 1st continuation)
// let b3 = vextq_u8(v, vn, 2);   // at each lane i, this holds byte[i+2]  (the 2nd continuation)
// Now at lane i you have v[i] (lead), b2[i] (cont1), b3[i] (cont2) all in the same lane, and every classification op after that is pure lane-wise:
// - 2-byte bitmap: index ci computed from v and b2, then vqtbl + vshl + vtst — all elementwise.
// - CJK range: vceq/vcge on v, b2, b3 — elementwise.
// Counting can potentially be done using xor operations to go form 100001000 to 011110000 -> fills
// the whole. this is potentially what we'll use in some of the fast fsm to detect boundary changes?
// checking ascii is just checking if there is a header:  0x8080808080808080 with 8 bytes. find the
// simd for this.
// UTF8 validator defines 12 different categories. SIMD JSOM only finds a few
// characters that are in a set, which is a very small set of ASCII (5 different characters) We
// need much more than that, we need to classify in one of the 16 different categories
// Every single byte has a different category. there are 256 possible bytes, and 12 categories for
// utf8 validators. SIMD registers take at least 16bytes.
//
