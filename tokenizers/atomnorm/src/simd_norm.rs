//! NEON accelerators (this module only compiles on aarch64) — pure prefix processors: each
//! consumes as much of the input as SIMD chunking allows and RETURNS ITS POSITION; the scalar
//! loops in `norm.rs`/`scan.rs` (the complete portable implementations) finish from there, which
//! makes the two paths byte-exact by construction. `STORE` write-through rides the caller's
//! reserved capacity slack.
use crate::norm::bmp_set;
use crate::tables::*;
use std::arch::aarch64::*;

const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Index of the first nonzero lane of a 0/FF mask (mask must be nonzero). `powv` is the
/// preloaded `POW` vector — callers hoist the load to kernel entry, off the stop path.
#[inline(always)]
unsafe fn first_lane(mask: uint8x16_t, powv: uint8x16_t) -> usize {
    // SAFETY (caller): plain NEON ops on values.
    unsafe {
        let m = vandq_u8(mask, powv);
        let mm = (vaddv_u8(vget_low_u8(m)) as u16) | ((vaddv_u8(vget_high_u8(m)) as u16) << 8);
        mm.trailing_zeros() as usize
    }
}

/// Byte classes of mixed [ascii | 2-byte] text: `(continuation, class_ok)` — a lane failing
/// `class_ok` (a 3/4-byte lead or invalid byte) ends the kernel's terrain.
#[inline(always)]
unsafe fn class2(v: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    unsafe {
        let ascii = vcltq_u8(v, vdupq_n_u8(0x80));
        let cont = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x80)), vcltq_u8(v, vdupq_n_u8(0xC0)));
        let lead2 = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xC2)), vcleq_u8(v, vdupq_n_u8(0xDF)));
        (cont, vorrq_u8(vorrq_u8(ascii, cont), lead2))
    }
}

/// A 2048-bit set (bitmap words reinterpreted as 256 `vqtbl` bytes), loaded once per kernel entry.
#[inline(always)]
unsafe fn load2048(p: *const u8) -> [uint8x16x4_t; 4] {
    unsafe {
        [
            vld1q_u8_x4(p),
            vld1q_u8_x4(p.add(64)),
            vld1q_u8_x4(p.add(128)),
            vld1q_u8_x4(p.add(192)),
        ]
    }
}

/// The 2-byte chars' set bits, evaluated at their continuation lanes entirely in 8-bit lanes:
/// `idx = (prev & 0x1F) << 3 | (cur & 0x3F) >> 3` picks the table byte, `cur & 7` the bit.
/// Returns a 0/FF mask (meaningful only at continuation lanes — mask with `cont`).
#[inline(always)]
unsafe fn probe2048(t: [uint8x16x4_t; 4], prev: uint8x16_t, v: uint8x16_t) -> uint8x16_t {
    unsafe {
        let idx = vorrq_u8(
            vshlq_n_u8::<3>(vandq_u8(prev, vdupq_n_u8(0x1F))),
            vshrq_n_u8::<3>(vandq_u8(v, vdupq_n_u8(0x3F))),
        );
        let byte = vorrq_u8(
            vorrq_u8(
                vqtbl4q_u8(t[0], idx),
                vqtbl4q_u8(t[1], vsubq_u8(idx, vdupq_n_u8(64))),
            ),
            vorrq_u8(
                vqtbl4q_u8(t[2], vsubq_u8(idx, vdupq_n_u8(128))),
                vqtbl4q_u8(t[3], vsubq_u8(idx, vdupq_n_u8(192))),
            ),
        );
        let sh = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(7))));
        let bit = vandq_u8(vshlq_u8(byte, sh), vdupq_n_u8(1));
        vtstq_u8(bit, bit)
    }
}

/// Uppercase `A..=Z` mask — false for every byte ≥ 0x80 (the subtract wraps far above 26).
#[inline(always)]
unsafe fn upper_mask(v: uint8x16_t) -> uint8x16_t {
    unsafe { vcltq_u8(vsubq_u8(v, vdupq_n_u8(b'A')), vdupq_n_u8(26)) }
}

/// The `CLEAN` ASCII policy masks: `(fold → ' ', removal)`. 1 = bert (`\t\n\r` fold; other
/// controls and DEL are removed), 2 = nmt (`\t\n\x0C\r` fold; controls minus NUL removed).
#[inline(always)]
unsafe fn ascii_policy<const CLEAN: u8>(v: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    unsafe {
        let eq = |b: u8| vceqq_u8(v, vdupq_n_u8(b));
        let f = match CLEAN {
            1 => vorrq_u8(vorrq_u8(eq(9), eq(10)), eq(13)),
            2 => vorrq_u8(vorrq_u8(eq(9), eq(10)), vorrq_u8(eq(12), eq(13))),
            _ => return (vdupq_n_u8(0), vdupq_n_u8(0)),
        };
        let mut ctl = vandq_u8(vcltq_u8(v, vdupq_n_u8(0x20)), vmvnq_u8(f));
        if CLEAN == 2 {
            ctl = vandq_u8(ctl, vmvnq_u8(eq(0))); // nmt keeps NUL
        }
        (f, vorrq_u8(ctl, eq(0x7F)))
    }
}

/// SIMD prefix of `norm::skip_clean` — stops at the first suspect lead or when < 32 bytes remain.
pub(crate) fn skip_clean<const STORE: bool>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
    mask: &[u8; 64],
) -> usize {
    let n = bytes.len();
    // SAFETY: loads gated by `i + 32 <= n`; stores by reserved capacity; stops are lead bytes.
    unsafe {
        let tbl = vld1q_u8_x4(mask.as_ptr());
        let powv = vld1q_u8(POW.as_ptr());
        let c0 = vdupq_n_u8(0xC0);
        let v_out = out.as_mut_vec();
        let mut len = v_out.len();
        // 32 bytes per iteration: the table output IS the hit mask (0/FF), one vmaxv for both chunks
        while i + 32 <= n {
            let a = vld1q_u8(bytes.as_ptr().add(i));
            let b = vld1q_u8(bytes.as_ptr().add(i + 16));
            if STORE {
                vst1q_u8(v_out.as_mut_ptr().add(len), a);
                vst1q_u8(v_out.as_mut_ptr().add(len + 16), b);
            }
            let ha = vqtbl4q_u8(tbl, vqsubq_u8(a, c0));
            let hb = vqtbl4q_u8(tbl, vqsubq_u8(b, c0));
            if vmaxvq_u8(vorrq_u8(ha, hb)) == 0 {
                if STORE {
                    len += 32;
                }
                i += 32;
                continue;
            }
            // locate the first suspect lane across the two chunks (no extra reduction: reuse the
            // POW-packed mask of chunk `a` for both the emptiness test and the position)
            let ma = vandq_u8(ha, powv);
            let mm =
                (vaddv_u8(vget_low_u8(ma)) as u16) | ((vaddv_u8(vget_high_u8(ma)) as u16) << 8);
            let k = if mm != 0 {
                mm.trailing_zeros() as usize
            } else {
                16 + first_lane(hb, powv)
            };
            if STORE {
                v_out.set_len(len + k);
            }
            return i + k;
        }
        if STORE {
            v_out.set_len(len);
        }
    }
    i
}

/// SIMD prefix of `norm::skip2_ascii` — returns at a set char, a non-(ascii|2-byte) boundary,
/// or when < 16 bytes remain; the scalar caller re-derives the classification at that position.
///
/// NOTE deliberately hand-inlined (not `class2`/`probe2048`): with one table the four `vqtbl4`
/// groups must stay individual SSA values — the helper-factored variant (array-of-tables, even
/// by value) measured +25% on this kernel's hottest path (NFD Russian).
pub(crate) fn skip2_ascii<const STORE: bool>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
) -> usize {
    let n = bytes.len();
    // SAFETY: loads gated by `i + 16 <= n`; stores ride reserved capacity; stops land on char starts
    // (a suspect continuation rolls back to its lead, which is always inside this chunk or the carry).
    unsafe {
        let bm = BMP_SET.as_ptr() as *const u8;
        let (t0, t1, t2, t3) = (
            vld1q_u8_x4(bm),
            vld1q_u8_x4(bm.add(64)),
            vld1q_u8_x4(bm.add(128)),
            vld1q_u8_x4(bm.add(192)),
        );
        let v_out = out.as_mut_vec();
        let mut len = v_out.len();
        let mut carry: u8 = 0;
        while i + 16 <= n {
            let v = vld1q_u8(bytes.as_ptr().add(i));
            if STORE {
                vst1q_u8(v_out.as_mut_ptr().add(len), v);
            }
            let ascii = vcltq_u8(v, vdupq_n_u8(0x80));
            let cont = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x80)), vcltq_u8(v, vdupq_n_u8(0xC0)));
            let lead2 = vandq_u8(vcgeq_u8(v, vdupq_n_u8(0xC2)), vcleq_u8(v, vdupq_n_u8(0xDF)));
            let class_ok = vorrq_u8(vorrq_u8(ascii, cont), lead2);
            // union-bit of each 2-byte char, evaluated at its continuation lane (8-bit throughout)
            let prev = vextq_u8::<15>(vdupq_n_u8(carry), v);
            let idx = vorrq_u8(
                vshlq_n_u8::<3>(vandq_u8(prev, vdupq_n_u8(0x1F))),
                vshrq_n_u8::<3>(vandq_u8(v, vdupq_n_u8(0x3F))),
            );
            let byte = vorrq_u8(
                vorrq_u8(
                    vqtbl4q_u8(t0, idx),
                    vqtbl4q_u8(t1, vsubq_u8(idx, vdupq_n_u8(64))),
                ),
                vorrq_u8(
                    vqtbl4q_u8(t2, vsubq_u8(idx, vdupq_n_u8(128))),
                    vqtbl4q_u8(t3, vsubq_u8(idx, vdupq_n_u8(192))),
                ),
            );
            let sh = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(7))));
            let bit = vandq_u8(vshlq_u8(byte, sh), vdupq_n_u8(1));
            let sus = vandq_u8(cont, vtstq_u8(bit, bit));
            let bad = vorrq_u8(vmvnq_u8(class_ok), sus);
            if vmaxvq_u8(bad) == 0 {
                carry = vgetq_lane_u8::<15>(v);
                if STORE {
                    len += 16;
                }
                i += 16;
                continue;
            }
            let mut m = [0u8; 16];
            vst1q_u8(m.as_mut_ptr(), bad);
            let k = m.iter().position(|&x| x != 0).unwrap();
            // a suspect continuation belongs to the char starting one byte earlier
            let back = usize::from(bytes[i + k] >= 0x80 && bytes[i + k] < 0xC0);
            let pos = i + k - back;
            if STORE {
                v_out.set_len(len + (k - back));
            }
            return pos; // scalar caller re-classifies the char here
        }
        if STORE {
            v_out.set_len(len);
        }
    }
    i
}

/// SIMD prefix of `scan::next_hit` — the scan normalizers' fused lane: one `vqtbl4` lead probe per
/// 16 bytes plus the in-register ASCII policy (`LOWER` flips `A..=Z` — the PR #2036 port — and
/// `CLEAN` folds whitespace to `' '` and stops at removal bytes). Check mode (`!STORE`) also stops
/// at any ASCII transform — the borrow gate; write mode stores the transformed bytes. `ASCII_SET`
/// (runtime sets) additionally probes ASCII membership. Stops are always char boundaries.
pub(crate) fn scan_prefix<
    const STORE: bool,
    const LOWER: bool,
    const CLEAN: u8,
    const ASCII_SET: bool,
>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
    lead: &[u8; 64],
    set: &[u64; 1024],
) -> usize {
    let n = bytes.len();
    // SAFETY: loads gated by `i + 16 <= n`; stores by the driver's reserved-capacity invariant;
    // `set_len` covers only verified bytes.
    unsafe {
        let tbl = vld1q_u8_x4(lead.as_ptr());
        let powv = vld1q_u8(POW.as_ptr());
        let c0 = vdupq_n_u8(0xC0);
        // runtime sets may contain ASCII members: their 128 bits live in the set's first 16 bytes
        let ascii_tbl = vld1q_u8(set.as_ptr() as *const u8);
        let v_out = out.as_mut_vec();
        let mut len = v_out.len();
        while i + 16 <= n {
            let v = vld1q_u8(bytes.as_ptr().add(i));
            let mut hit = vqtbl4q_u8(tbl, vqsubq_u8(v, c0));
            if ASCII_SET {
                // bytes ≥ 0x80 index past the 16-byte table → 0, so only ASCII lanes can fire
                let byte = vqtbl1q_u8(ascii_tbl, vshrq_n_u8::<3>(v));
                let sh = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(7))));
                let bit = vandq_u8(vshlq_u8(byte, sh), vdupq_n_u8(1));
                hit = vorrq_u8(hit, vtstq_u8(bit, bit));
            }
            let upper = if LOWER { upper_mask(v) } else { vdupq_n_u8(0) };
            let (fold, rm) = ascii_policy::<CLEAN>(v);
            let stop = if STORE {
                vorrq_u8(hit, rm)
            } else {
                vorrq_u8(vorrq_u8(hit, rm), vorrq_u8(upper, fold))
            };
            if STORE {
                let mut t = v;
                if LOWER {
                    t = vbslq_u8(upper, vorrq_u8(v, vdupq_n_u8(0x20)), t);
                }
                if CLEAN != 0 {
                    t = vbslq_u8(fold, vdupq_n_u8(b' '), t);
                }
                vst1q_u8(v_out.as_mut_ptr().add(len), t);
            }
            if vmaxvq_u8(stop) == 0 {
                if STORE {
                    len += 16;
                }
                i += 16;
                continue;
            }
            let k = first_lane(stop, powv);
            if STORE {
                v_out.set_len(len + k);
            }
            return i + k;
        }
        if STORE {
            v_out.set_len(len);
        }
    }
    i
}

/// The two-table case-swap kernel — write-mode only, for case-folding scans riding 2-byte cased
/// scripts (Cyrillic/Greek/Latin-1). 16 mixed [ascii | 2-byte] bytes per iteration: the ASCII
/// lanes get the `|0x20` + `CLEAN`-fold policy; each 2-byte char is probed IN-REGISTER against
/// two 2048-bit tables — the scan SET (stop → scalar fixup) and the case-swap SOURCE (`reg2`:
/// uppercase mapping to exactly `cp + 0x20`). Source hits are transformed in place: the TARGET is
/// arithmetic — continuation `+0x20`, with the UTF-8 carry (`> 0xBF`) folding back `-0x40` and
/// adding `+1` to the lead lane (Р D0A0 → р D180). A trailing lead byte is left for the next
/// iteration so a pair never straddles a chunk. Returns its stop position (a char boundary).
pub(crate) fn scan2_case<const CLEAN: u8>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
    set: &[u64; 1024],
    reg2: &[u64; 32],
) -> usize {
    let n = bytes.len();
    // SAFETY: loads gated by `i + 16 <= n`; stores by the driver's reserved-capacity invariant;
    // `set_len` covers only verified bytes; stops roll back to char starts.
    unsafe {
        let st = load2048(set.as_ptr() as *const u8);
        let rt = load2048(reg2.as_ptr() as *const u8);
        let powv = vld1q_u8(POW.as_ptr());
        let zero = vdupq_n_u8(0);
        let v_out = out.as_mut_vec();
        let mut len = v_out.len();
        while i + 16 <= n {
            let v = vld1q_u8(bytes.as_ptr().add(i));
            let (cont, class_ok) = class2(v);
            // lane 0 never starts mid-pair (entry and advance are char boundaries), so prev=0 is fine
            let prev = vextq_u8::<15>(zero, v);
            let set_hit = vandq_u8(cont, probe2048(st, prev, v));
            let reg_hit = vandq_u8(cont, probe2048(rt, prev, v));
            let upper = upper_mask(v);
            let (fold, rm) = ascii_policy::<CLEAN>(v);
            let bad = vorrq_u8(vorrq_u8(vmvnq_u8(class_ok), vbicq_u8(set_hit, reg_hit)), rm);
            // transform: ASCII policy, then the case-swap target on source-hit lanes
            let mut t = vbslq_u8(upper, vorrq_u8(v, vdupq_n_u8(0x20)), v);
            if CLEAN != 0 {
                t = vbslq_u8(fold, vdupq_n_u8(b' '), t);
            }
            let c1 = vaddq_u8(v, vdupq_n_u8(0x20));
            let ovf = vandq_u8(vcgtq_u8(c1, vdupq_n_u8(0xBF)), reg_hit);
            t = vbslq_u8(
                reg_hit,
                vbslq_u8(ovf, vsubq_u8(c1, vdupq_n_u8(0x40)), c1),
                t,
            );
            // the overflowing pair's LEAD (one lane earlier) gets +1
            t = vaddq_u8(t, vandq_u8(vextq_u8::<1>(ovf, zero), vdupq_n_u8(1)));
            vst1q_u8(v_out.as_mut_ptr().add(len), t);
            if vmaxvq_u8(bad) == 0 {
                // never consume a trailing lead: its continuation transforms next iteration
                let adv = 16 - usize::from((0xC2..=0xDF).contains(&bytes[i + 15]));
                len += adv;
                i += adv;
                continue;
            }
            let k = first_lane(bad, powv);
            // a bad continuation belongs to the char starting one byte earlier
            let back = usize::from(bytes[i + k] >= 0x80 && bytes[i + k] < 0xC0);
            v_out.set_len(len + (k - back));
            return i + k - back;
        }
        v_out.set_len(len);
    }
    i
}

/// SIMD prefix of `norm::skip3` — returns at a set char / width change / when < 48 bytes remain.
pub(crate) fn skip3<const STORE: bool>(bytes: &[u8], mut i: usize, out: &mut String) -> usize {
    let n = bytes.len();
    // SAFETY: loads gated by `i + 48 <= n`; stores by reserved capacity (`+48` slack); `set_len`
    // only covers verified whole chars.
    unsafe {
        let v_out = out.as_mut_vec();
        let mut len = v_out.len();
        while i + 48 <= n {
            let (mut cps, mut lok) = ([0u16; 16], [0u8; 16]);
            let x = vld3q_u8(bytes.as_ptr().add(i));
            if STORE {
                vst3q_u8(v_out.as_mut_ptr().add(len), x);
            }
            let ok = vandq_u8(
                vcgeq_u8(x.0, vdupq_n_u8(0xE0)),
                vcleq_u8(x.0, vdupq_n_u8(0xEF)),
            );
            vst1q_u8(lok.as_mut_ptr(), ok);
            for (h, (l8, b18, b28)) in [
                (
                    0usize,
                    (vget_low_u8(x.0), vget_low_u8(x.1), vget_low_u8(x.2)),
                ),
                (8, (vget_high_u8(x.0), vget_high_u8(x.1), vget_high_u8(x.2))),
            ] {
                let l = vandq_u16(vmovl_u8(l8), vdupq_n_u16(0x0F));
                let b1 = vandq_u16(vmovl_u8(b18), vdupq_n_u16(0x3F));
                let b2 = vandq_u16(vmovl_u8(b28), vdupq_n_u16(0x3F));
                let cp = vorrq_u16(vorrq_u16(vshlq_n_u16::<12>(l), vshlq_n_u16::<6>(b1)), b2);
                vst1q_u16(cps.as_mut_ptr().add(h), cp);
            }
            match (0..16).position(|l| lok[l] != 0xFF || bmp_set(cps[l])) {
                Some(l) => {
                    if STORE {
                        v_out.set_len(len + l * 3);
                    }
                    return i + l * 3; // scalar caller re-classifies
                }
                None => {
                    if STORE {
                        len += 48;
                    }
                    i += 48;
                }
            }
        }
        if STORE {
            v_out.set_len(len);
        }
    }
    i
}
