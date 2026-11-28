//! RVV byte matching and bitstream construction for 64-byte blocks.
use crate::{AUX_NONE, AUX_SLASH, Blk, lead_run};
use core::arch::asm;

/// Fold 64 tags through the grammar LUT and extract filled code bit-planes.
///
/// # Safety
/// RVV must be available, both slices must contain `base..base + 64`, and tags must be below 64.
#[inline]
pub(crate) unsafe fn build64<const AUX: u8, const P3: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    lut: &[u8; 64],
    cur_code: u8,
    cur_aux: bool,
) -> (Blk, u8) {
    let mut b = Blk::default();
    let last: usize;
    let vtype: usize;
    // Use the smallest whole-register group holding the 64-byte LUT. VL stays 64,
    // so every mask store writes exactly eight bytes at every supported VLEN.
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            "csrr {width}, vlenb",
            "li {vtype}, 192", // e8, m1, ta, ma
            "2:",
            "bgeu {width}, {len}, 3f",
            "slli {width}, {width}, 1",
            "addi {vtype}, {vtype}, 1",
            "j 2b",
            "3:",
            "vsetvl zero, {len}, {vtype}",
            "vle8.v v8, ({lut})",
            "vle8.v v12, ({tags})",
            "vrgather.vv v16, v8, v12",
            "vmseq.vi v0, v12, 15",
            "vsm.v v0, ({cont})",
            // Fill from the preceding byte, then from two bytes back. The second step
            // resolves the two remaining continuations of a four-byte character.
            "vslide1up.vx v20, v16, {seed}",
            "vmerge.vvm v16, v16, v20, v0",
            "vmseq.vi v0, v16, 7",
            "vmv.v.x v20, {seed}",
            "vslideup.vi v20, v16, 2",
            "vmerge.vvm v16, v16, v20, v0",
            "vslidedown.vx v20, v16, {last_lane}",
            "vmv.x.s {last}, v20",
            "vand.vi v20, v16, 1",
            "vmsne.vi v0, v20, 0",
            "vsm.v v0, ({p0})",
            "vand.vi v20, v16, 2",
            "vmsne.vi v0, v20, 0",
            "vsm.v v0, ({p1})",
            "vand.vi v20, v16, 4",
            "vmsne.vi v0, v20, 0",
            "vsm.v v0, ({p2})",
            ".if {p3_enabled}",
            "vand.vi v20, v16, 8",
            "vmsne.vi v0, v20, 0",
            "vsm.v v0, ({p3})",
            ".endif",
            ".option pop",
            len = in(reg) 64usize,
            last_lane = in(reg) 63usize,
            seed = in(reg) cur_code as usize,
            lut = in(reg) lut.as_ptr(),
            tags = in(reg) tags.as_ptr().add(base),
            cont = in(reg) &mut b.cont,
            p0 = in(reg) &mut b.p0,
            p1 = in(reg) &mut b.p1,
            p2 = in(reg) &mut b.p2,
            p3 = in(reg) &mut b.p3,
            p3_enabled = const P3 as u8,
            last = out(reg) last,
            width = out(reg) _, vtype = out(reg) vtype,
            out("v0") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            options(nostack),
        );
    }
    b.cont = u64::from_le(b.cont);
    b.p0 = u64::from_le(b.p0);
    b.p1 = u64::from_le(b.p1);
    b.p2 = u64::from_le(b.p2);
    b.p3 = u64::from_le(b.p3);
    if AUX == AUX_NONE {
        return (b, last as u8);
    }
    if AUX == AUX_SLASH {
        b.aux = unsafe { eq64(text, base, b'/') };
        return (b, last as u8);
    }
    // The CJK ranges are all three-byte characters. Lookahead is bounded independently
    // of the full-block loads, including when the final character crosses a block edge.
    let next0 = text.get(base + 64).copied().unwrap_or(0);
    let next1 = text.get(base + 65).copied().unwrap_or(0);
    let mut leads = 0u64;
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            "vsetvl zero, {len}, {vtype}",
            "vle8.v v8, ({src})",
            "vslide1down.vx v12, v8, {next0}",
            "vslide1down.vx v16, v12, {next1}",
            // Hiragana/Katakana: E3 [81-83] xx.
            "vsub.vx v20, v12, {lo81}",
            "vmsleu.vi v0, v20, 2",
            "vmseq.vx v1, v8, {e3}",
            "vmand.mm v0, v0, v1",
            // Han: E4 [B8-BF] xx, E5-E8 xx xx, E9 [80-BD] xx or E9 BE [80-A5].
            "vmsgeu.vx v1, v12, {b8}",
            "vmseq.vx v2, v8, {e4}",
            "vmand.mm v1, v1, v2",
            "vmor.mm v0, v0, v1",
            "vsub.vx v20, v8, {e5}",
            "vmsleu.vi v1, v20, 3",
            "vmor.mm v0, v0, v1",
            "vmsleu.vx v1, v12, {bd}",
            "vmseq.vx v2, v12, {be}",
            "vmsleu.vx v3, v16, {a5}",
            "vmand.mm v2, v2, v3",
            "vmor.mm v1, v1, v2",
            "vmseq.vx v2, v8, {e9}",
            "vmand.mm v1, v1, v2",
            "vmor.mm v0, v0, v1",
            "vsm.v v0, ({dst})",
            ".option pop",
            len = in(reg) 64usize,
            vtype = in(reg) vtype,
            src = in(reg) text.as_ptr().add(base),
            dst = in(reg) &mut leads,
            next0 = in(reg) next0 as usize, next1 = in(reg) next1 as usize,
            lo81 = in(reg) 0x81usize, e3 = in(reg) 0xe3usize,
            b8 = in(reg) 0xb8usize, e4 = in(reg) 0xe4usize,
            e5 = in(reg) 0xe5usize, bd = in(reg) 0xbdusize,
            be = in(reg) 0xbeusize, a5 = in(reg) 0xa5usize, e9 = in(reg) 0xe9usize,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            options(nostack),
        );
    }
    let mut leads = u64::from_le(leads);
    let lim = text.len().saturating_sub(base + 2);
    if lim < 64 {
        leads &= (1u64 << lim) - 1;
    }
    b.aux = leads | (leads << 1) | (leads << 2);
    if cur_aux {
        b.aux |= lead_run(b.cont, !0);
    }
    (b, last as u8)
}

/// Compare 64 bytes with `b`, returning one bit per byte.
///
/// # Safety
/// RVV must be available and `base + 64 <= text.len()`.
#[inline]
pub(crate) unsafe fn eq64(text: &[u8], base: usize, b: u8) -> u64 {
    let mut mask = 0u64;
    // RVV guarantees VLEN >= 128: e8/m4 holds at least 64 bytes.
    // vsm writes the 64 comparison bits in byte order, including on unaligned input.
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            "vsetvli zero, {len}, e8, m4, ta, ma",
            "vle8.v v8, ({src})",
            "vmseq.vx v0, v8, {byte}",
            "vsm.v v0, ({dst})",
            ".option pop",
            len = in(reg) 64usize,
            src = in(reg) text.as_ptr().add(base),
            byte = in(reg) b as usize,
            dst = in(reg) &mut mask,
            out("v0") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
    u64::from_le(mask)
}

/// Test whether any of the 64 bytes equals `b`.
///
/// # Safety
/// RVV must be available and `base + 64 <= text.len()`.
#[inline]
pub(crate) unsafe fn any64(text: &[u8], base: usize, b: u8) -> bool {
    let first: isize;
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            "vsetvli zero, {len}, e8, m4, ta, ma",
            "vle8.v v8, ({src})",
            "vmseq.vx v0, v8, {byte}",
            "vfirst.m {first}, v0",
            ".option pop",
            len = in(reg) 64usize,
            src = in(reg) text.as_ptr().add(base),
            byte = in(reg) b as usize,
            first = lateout(reg) first,
            out("v0") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack, readonly),
        );
    }
    first >= 0
}
