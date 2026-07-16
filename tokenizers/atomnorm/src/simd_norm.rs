//! NEON accelerators for the three skip kernels — pure prefix processors: each consumes as much of
//! the input as SIMD chunking allows and RETURNS ITS POSITION; the scalar loops in `norm.rs` (the
//! complete portable implementation) finish from there, which makes the two paths byte-exact by
//! construction. `STORE` write-through rides the caller's reserved capacity slack.
use crate::norm::bmp_set;
use crate::tables::*;

/// SIMD prefix of `norm::skip_clean` — stops at the first suspect lead or when < 32 bytes remain.
pub(crate) fn skip_clean<const STORE: bool>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
    mask: &[u8; 64],
) -> usize {
    let n = bytes.len();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: loads gated by `i + 32 <= n`; stores by reserved capacity; stops are lead bytes.
    unsafe {
        use std::arch::aarch64::*;
        const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
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
            // locate the first suspect lane across the two chunks
            let ma = vandq_u8(ha, powv);
            let mm =
                (vaddv_u8(vget_low_u8(ma)) as u16) | ((vaddv_u8(vget_high_u8(ma)) as u16) << 8);
            let k = if mm != 0 {
                mm.trailing_zeros() as usize
            } else {
                let mb = vandq_u8(hb, powv);
                let m2 =
                    (vaddv_u8(vget_low_u8(mb)) as u16) | ((vaddv_u8(vget_high_u8(mb)) as u16) << 8);
                16 + m2.trailing_zeros() as usize
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
pub(crate) fn skip2_ascii<const STORE: bool>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
) -> usize {
    let n = bytes.len();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: loads gated by `i + 16 <= n`; stores ride reserved capacity; stops land on char starts
    // (a suspect continuation rolls back to its lead, which is always inside this chunk or the carry).
    unsafe {
        use std::arch::aarch64::*;
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

/// SIMD prefix of `norm::skip3` — returns at a set char / width change / when < 48 bytes remain.
pub(crate) fn skip3<const STORE: bool>(bytes: &[u8], mut i: usize, out: &mut String) -> usize {
    let n = bytes.len();
    #[cfg(target_arch = "aarch64")]
    // SAFETY: loads gated by `i + 16*W <= n`; stores by reserved capacity (`+48` slack); `set_len`
    // only covers verified whole chars.
    unsafe {
        use std::arch::aarch64::*;
        let v_out = out.as_mut_vec();
        let mut len = v_out.len();
        while i + 48 <= n {
            let (mut cps, mut lok) = ([0u16; 16], [0u8; 16]);
            if false {
                let x = vld2q_u8(bytes.as_ptr().add(i));
                if STORE {
                    vst2q_u8(v_out.as_mut_ptr().add(len), x);
                }
                let ok = vandq_u8(
                    vcgeq_u8(x.0, vdupq_n_u8(0xC2)),
                    vcleq_u8(x.0, vdupq_n_u8(0xDF)),
                );
                vst1q_u8(lok.as_mut_ptr(), ok);
                for (h, (l8, c8)) in [
                    (0usize, (vget_low_u8(x.0), vget_low_u8(x.1))),
                    (8, (vget_high_u8(x.0), vget_high_u8(x.1))),
                ] {
                    let l = vandq_u16(vmovl_u8(l8), vdupq_n_u16(0x1F));
                    let cc = vandq_u16(vmovl_u8(c8), vdupq_n_u16(0x3F));
                    vst1q_u16(cps.as_mut_ptr().add(h), vorrq_u16(vshlq_n_u16::<6>(l), cc));
                }
            } else {
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
