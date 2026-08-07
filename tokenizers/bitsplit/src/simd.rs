//! NEON block builder: 64 tag bytes + 64 text bytes → the 10 class bitstreams of one [`Blk`].
//!
//! This is the step the paper spends its GPU budget on too — turning bytes into bitstreams. The
//! shape here is a 64×8 bit-matrix transpose done 8 lanes at a time: weight each compare result by
//! its lane's power of two (`POW`) and let three `vpaddq_u8` rounds fold 64 lanes into the 8 bytes
//! of one `u64`. That is ~9 ops per stream per 64 bytes, versus 4 separate 16-bit movemasks.
//!
//! Continuation bytes are resolved **before** extraction (≤3 `vext`+`vbsl`, the same trick
//! `atomsplit::simd_fsm` uses), so every stream comes out *filled* — a multi-byte char sets its bit
//! on all of its bytes. That is what lets the bitstream program read "previous char's class" as a
//! plain `<< 1` with no char-width arithmetic.

use crate::{AUX_HAN, AUX_NONE, AUX_SLASH, Blk, lead_run};
use core::arch::aarch64::*;

const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// 64 lanes of `0x00`/`0xFF` → one `u64`. Three `vpaddq_u8` rounds: lane pairs, then quads, then
/// octets — after which the low 8 bytes of the result are the 8 bytes of the mask, in order.
#[inline(always)]
unsafe fn mm64(v: [uint8x16_t; 4], pow: uint8x16_t) -> u64 {
    unsafe {
        let ab = vpaddq_u8(vandq_u8(v[0], pow), vandq_u8(v[1], pow));
        let cd = vpaddq_u8(vandq_u8(v[2], pow), vandq_u8(v[3], pow));
        let x = vpaddq_u8(ab, cd);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(x, x)))
    }
}

/// Build one **full** 64-byte block, folding tags through `lut` into dense codes.
/// Build one **full** 64-byte block. `cur_code` / `cur_cjk` describe the byte just before it, so a
/// block opening mid-char keeps inheriting its lead's class. Returns the block's last filled code.
///
/// # Safety
/// `base + 64 <= tags.len()` and `base + 64 <= text.len()`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn build64<const AUX: u8, const P3: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    lut: &[u8; 64],
    cur_code: u8,
    cur_aux: bool,
) -> (Blk, u8) {
    unsafe {
        let pow = vld1q_u8(POW.as_ptr());
        let seven = vdupq_n_u8(7);
        let tbl = uint8x16x4_t(
            vld1q_u8(lut.as_ptr()),
            vld1q_u8(lut.as_ptr().add(16)),
            vld1q_u8(lut.as_ptr().add(32)),
            vld1q_u8(lut.as_ptr().add(48)),
        );

        // ── tags → dense codes, then fill the continuation lanes from the left so every lane
        // carries its char's code. Two steps suffice, not three: after shifting by 1 every lane is
        // correct at distance 1, so shifting *that* by 2 covers distances 2 and 3 — which is the
        // most a 4-byte char can need.
        let mut isc = [vdupq_n_u8(0); 4];
        let mut cd = [vdupq_n_u8(0); 4];
        let mut prev = vdupq_n_u8(cur_code);
        for k in 0..4 {
            let raw = vqtbl4q_u8(tbl, vld1q_u8(tags.as_ptr().add(base + k * 16)));
            isc[k] = vceqq_u8(raw, seven);
            let c = vbslq_u8(isc[k], vextq_u8::<15>(prev, raw), raw);
            let c = vbslq_u8(vceqq_u8(c, seven), vextq_u8::<14>(prev, c), c);
            prev = c;
            cd[k] = c;
        }
        let last_code = vgetq_lane_u8::<15>(prev);
        // ── 3 bit-planes of the code. Every stream is a boolean function of these (`decode`), so
        // this is 3 extractions where one-hot class bits needed 6 — and `cont` comes from `isc`,
        // which the fill needed anyway.
        let plane = |bit: u8| {
            let d = vdupq_n_u8(bit);
            mm64(
                [
                    vtstq_u8(cd[0], d),
                    vtstq_u8(cd[1], d),
                    vtstq_u8(cd[2], d),
                    vtstq_u8(cd[3], d),
                ],
                pow,
            )
        };

        let mut b = Blk {
            cont: mm64(isc, pow),
            p0: plane(1),
            p1: plane(2),
            p2: plane(4),
            p3: if P3 { plane(8) } else { 0 },
            aux: 0,
        };
        if AUX == AUX_NONE {
            return (b, last_code);
        }
        // ── text is loaded only for the aux (text-derived) stream.
        let ntext = text.len();
        let tv = [
            vld1q_u8(text.as_ptr().add(base)),
            vld1q_u8(text.as_ptr().add(base + 16)),
            vld1q_u8(text.as_ptr().add(base + 32)),
            vld1q_u8(text.as_ptr().add(base + 48)),
        ];

        if AUX == AUX_SLASH {
            // single ASCII byte — no fill needed, `/` is its own char
            let sl = vdupq_n_u8(b'/');
            b.aux = mm64(
                [
                    vceqq_u8(tv[0], sl),
                    vceqq_u8(tv[1], sl),
                    vceqq_u8(tv[2], sl),
                    vceqq_u8(tv[3], sl),
                ],
                pow,
            );
            return (b, last_code);
        }
        if AUX == AUX_HAN {
            // ponytail: scalar Han range test; vectorise like the CJK path below if kimi ever
            // shows up on a throughput bench.
            let lim = ntext.min(base + 64);
            let mut leads = 0u64;
            for p in base..lim {
                if tags[p] != crate::CONT && crate::han::is_han_at(text, p) {
                    leads |= 1u64 << (p - base);
                }
            }
            b.aux = leads | (leads << 1) | (leads << 2);
            if cur_aux {
                b.aux |= lead_run(b.cont, !0);
            }
            return (b, last_code);
        }

        // ── the CJK range test lives in the raw bytes, not the tags. It is a 3-byte predicate, so
        // it runs in vector space on `vext`-aligned b1/b2 and extracts ONE mask — testing the 9
        // byte predicates as separate bitstreams costs 9 extractions for the same answer. A
        // `vmaxvq` over the lead range gates the whole thing away on Latin/code for ~8 ops.
        let e3e9 = |v| vcleq_u8(vsubq_u8(v, vdupq_n_u8(0xE3)), vdupq_n_u8(6));
        let any = vmaxvq_u8(vorrq_u8(
            vorrq_u8(e3e9(tv[0]), e3e9(tv[1])),
            vorrq_u8(e3e9(tv[2]), e3e9(tv[3])),
        ));
        if any != 0 {
            // b1/b2 of a char at lane 15 come from the next chunk — and for the last chunk, from
            // the next block, which may not exist.
            let tail = {
                let mut buf = [0u8; 16];
                let off = base + 64;
                let avail = text.len().saturating_sub(off).min(16);
                buf[..avail].copy_from_slice(&text[off..off + avail]);
                vld1q_u8(buf.as_ptr())
            };
            // Hiragana/Katakana U+3040..30FF is exactly E3 [81-83] xx; Han U+4E00..9FA5 is
            // E4 [B8-BF] xx / E5-E8 xx xx / E9 [80-BD] xx / E9 BE [80-A5].
            let cjkv = |v: uint8x16_t, nx: uint8x16_t| {
                let (b1, b2) = (vextq_u8::<1>(v, nx), vextq_u8::<2>(v, nx));
                let e9 = vandq_u8(
                    vceqq_u8(v, vdupq_n_u8(0xE9)),
                    vorrq_u8(
                        vcltq_u8(b1, vdupq_n_u8(0xBE)),
                        vandq_u8(
                            vceqq_u8(b1, vdupq_n_u8(0xBE)),
                            vcleq_u8(b2, vdupq_n_u8(0xA5)),
                        ),
                    ),
                );
                vorrq_u8(
                    vorrq_u8(
                        vandq_u8(
                            vceqq_u8(v, vdupq_n_u8(0xE3)),
                            vcleq_u8(vsubq_u8(b1, vdupq_n_u8(0x81)), vdupq_n_u8(2)),
                        ),
                        vandq_u8(
                            vceqq_u8(v, vdupq_n_u8(0xE4)),
                            vcgeq_u8(b1, vdupq_n_u8(0xB8)),
                        ),
                    ),
                    vorrq_u8(vcleq_u8(vsubq_u8(v, vdupq_n_u8(0xE5)), vdupq_n_u8(3)), e9),
                )
            };
            let mut leads = mm64(
                [
                    cjkv(tv[0], tv[1]),
                    cjkv(tv[1], tv[2]),
                    cjkv(tv[2], tv[3]),
                    cjkv(tv[3], tail),
                ],
                pow,
            );
            // `fsm_deepseek` reads 3 bytes unconditionally; refuse to classify a truncated tail.
            let lim = ntext.saturating_sub(base + 2);
            if lim < 64 {
                leads &= (1u64 << lim) - 1;
            }
            // every CJK char is 3 bytes → fill by two shifts; a char cut by the block edge is
            // picked up on the other side by `cur_aux` (its continuation bytes lead that block).
            b.aux = leads | (leads << 1) | (leads << 2);
        }
        if cur_aux {
            b.aux |= lead_run(b.cont, !0);
        }
        (b, last_code)
    }
}
