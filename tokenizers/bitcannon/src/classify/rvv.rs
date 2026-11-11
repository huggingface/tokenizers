//! RVV atom classification with shared sparse Unicode tables.
use super::{ATOM_TABLES, classify_scalar};
use core::arch::asm;

// Indexed byte loads address the two halves of each mixed block consecutively.
const _: () = {
    type Block = ([u8; 64], [u8; 64]);
    assert!(core::mem::size_of::<Block>() == 128);
    assert!(core::mem::offset_of!(Block, 0) == 0);
    assert!(core::mem::offset_of!(Block, 1) == 64);
    assert!(ATOM_TABLES.fast3_mixed.len() <= 512);
};

static ASCII: [u8; 128] = {
    let mut table = [0; 128];
    let mut i = 0;
    while i < 64 {
        table[i] = ATOM_TABLES.ascii_lo[i];
        table[i + 64] = ATOM_TABLES.ascii_hi[i];
        i += 1;
    }
    table
};

/// Classify a leading ASCII run without reading beyond the input slice.
#[inline]
unsafe fn ascii_prefix(text: &[u8], tags: &mut [u8]) -> usize {
    let mut remaining = text.len();
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            // Choose the smallest register group holding the 128-byte table.
            "csrr {width}, vlenb",
            "li {vtype}, 192", // e8, m1, ta, ma
            "5:",
            "bgeu {width}, {table_len}, 6f",
            "slli {width}, {width}, 1",
            "addi {vtype}, {vtype}, 1",
            "j 5b",
            "6:",
            "vsetvl zero, {table_len}, {vtype}",
            "vle8.v v8, ({table})",
            "2:",
            "vsetvl {vl}, {remaining}, {vtype}",
            "vle8.v v16, ({src})",
            "vmslt.vx v0, v16, zero",
            "vfirst.m {first}, v0",
            "bltz {first}, 3f",
            "vsetvl {vl}, {first}, {vtype}",
            "3:",
            "vrgather.vv v24, v8, v16",
            "vse8.v v24, ({dst})",
            "sub {remaining}, {remaining}, {vl}",
            "add {src}, {src}, {vl}",
            "add {dst}, {dst}, {vl}",
            "bgez {first}, 4f",
            "bnez {remaining}, 2b",
            "4:",
            ".option pop",
            table_len = in(reg) 128usize,
            table = in(reg) ASCII.as_ptr(),
            src = inout(reg) text.as_ptr() => _,
            dst = inout(reg) tags.as_mut_ptr() => _,
            remaining = inout(reg) remaining,
            vl = out(reg) _, first = out(reg) _,
            width = out(reg) _, vtype = out(reg) _,
            out("v0") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
        );
    }
    text.len() - remaining
}

// The byte vectors have the same lane count as their e16/m2 index vectors.
// Masked indexed loads reuse the shared sparse Unicode tables.
unsafe fn mixed(text: &[u8], tags: &mut [u8]) -> usize {
    let vl: usize;
    let fix: isize;
    unsafe {
        asm!(
            ".option push", ".option arch, +v",
            "vsetvli {vl}, {len}, e8, m1, tu, mu",
            "vle8.v v8, ({src})",
            "addi {tmp}, {src}, 1", "vle8.v v9, ({tmp})",
            "addi {tmp}, {src}, 2", "vle8.v v10, ({tmp})",
            "vmv.v.i v12, 15",
            // Wide vectors can keep the entire ASCII table in one register.
            "li {tmp}, 128", "vmsltu.vx v0, v8, {tmp}",
            "csrr {tmp}, vlenb", "li {width}, 128", "bltu {tmp}, {width}, 5f",
            "vsetvli zero, {width}, e8, m1, tu, mu", "vle8.v v24, ({ascii})",
            "vsetvli zero, {vl}, e8, m1, tu, mu", "vrgather.vv v12, v24, v8, v0.t", "j 6f",
            "5:",
            "vsetvli zero, {vl}, e16, m2, tu, mu", "vzext.vf2 v16, v8",
            "vsetvli zero, {vl}, e8, m1, tu, mu", "vluxei16.v v12, ({ascii}), v16, v0.t",
            "6:",
            // Resolve the same common Han range used by the NEON/x86 kernels.
            "li {tmp}, 234", "vmsltu.vx v2, v8, {tmp}",
            "li {tmp}, 228", "vmsgeu.vx v1, v8, {tmp}", "vmand.mm v2, v2, v1",
            "vmseq.vx v3, v8, {tmp}", "li {tmp}, 183", "vmseq.vx v4, v9, {tmp}",
            "vmand.mm v3, v3, v4", "vmandn.mm v2, v2, v3",
            "vmmv.m v0, v2", "li {tmp}, 48", "vmerge.vxm v12, v12, {tmp}, v0",
            // Two-byte characters, indexed by their eleven-bit codepoint.
            "li {tmp}, 224", "vmsltu.vx v0, v8, {tmp}",
            "li {tmp}, 192", "vmsgeu.vx v1, v8, {tmp}", "vmand.mm v0, v0, v1",
            "vcpop.m {tmp}, v0", "beqz {tmp}, 2f",
            "vsetvli zero, {vl}, e16, m2, tu, mu",
            "vzext.vf2 v16, v8", "li {tmp}, 31", "vand.vx v16, v16, {tmp}", "vsll.vi v16, v16, 6",
            "vzext.vf2 v18, v9", "li {tmp}, 63", "vand.vx v18, v18, {tmp}", "vor.vv v16, v16, v18",
            "vsetvli zero, {vl}, e8, m1, tu, mu",
            "vluxei16.v v12, ({two}), v16, v0.t",
            "2:",
            // Three-byte characters, including CJK: uniform blocks need one lookup.
            "li {tmp}, 240", "vmsltu.vx v0, v8, {tmp}",
            "li {tmp}, 224", "vmsgeu.vx v1, v8, {tmp}", "vmand.mm v0, v0, v1", "vmandn.mm v0, v0, v2",
            "vcpop.m {tmp}, v0", "beqz {tmp}, 4f",
            "vsetvli zero, {vl}, e16, m2, tu, mu",
            "vzext.vf2 v16, v8", "vand.vi v16, v16, 15", "vsll.vi v16, v16, 5",
            "vzext.vf2 v18, v9", "vsrl.vi v18, v18, 1", "li {tmp}, 31", "vand.vx v18, v18, {tmp}", "vor.vv v16, v16, v18",
            "vsetvli zero, {vl}, e8, m1, tu, mu",
            "vmv.v.i v13, -1", "vluxei16.v v13, ({uni}), v16, v0.t",
            "vmerge.vvm v12, v12, v13, v0",
            // Mixed blocks use the existing slot table and 128-byte subtables.
            "li {tmp}, 255", "vmseq.vx v1, v13, {tmp}", "vmand.mm v0, v0, v1", "vcpop.m {tmp}, v0", "beqz {tmp}, 4f",
            "vsetvli zero, {vl}, e16, m2, tu, mu",
            "vsll.vi v16, v16, 1", "vluxei16.v v18, ({slots}), v16, v0.t",
            "vsll.vi v18, v18, 7",
            "vzext.vf2 v16, v9", "vand.vi v16, v16, 1", "vsll.vi v16, v16, 6", "vor.vv v18, v18, v16",
            "vzext.vf2 v16, v10", "li {tmp}, 63", "vand.vx v16, v16, {tmp}", "vor.vv v18, v18, v16",
            "vsetvli zero, {vl}, e8, m1, tu, mu",
            "vluxei16.v v12, ({three}), v18, v0.t",
            "4:",
            // Four-byte leads are fixed up using the shared scalar table reader.
            "li {tmp}, 240", "vmsgeu.vx v0, v8, {tmp}", "vmerge.vim v12, v12, 14, v0",
            "vse8.v v12, ({dst})",
            "vmseq.vi v0, v12, 14", "vfirst.m {fix}, v0",
            ".option pop",
            vl = out(reg) vl, fix = out(reg) fix, tmp = out(reg) _, width = out(reg) _,
            len = in(reg) text.len() - 2, src = in(reg) text.as_ptr(), dst = in(reg) tags.as_mut_ptr(),
            ascii = in(reg) ASCII.as_ptr(), two = in(reg) ATOM_TABLES.group_tables.as_ptr(),
            uni = in(reg) ATOM_TABLES.fast3_uni.as_ptr(), slots = in(reg) ATOM_TABLES.fast3_slot.as_ptr(),
            three = in(reg) ATOM_TABLES.fast3_mixed.as_ptr(),
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v8") _, out("v9") _, out("v10") _,
            out("v12") _, out("v13") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v24") _,
            options(nostack),
        );
    }
    if fix >= 0 {
        for (j, tag) in tags[..vl].iter_mut().enumerate().skip(fix as usize) {
            if *tag == super::MB {
                *tag = ATOM_TABLES.classify_char(text, j);
            }
        }
    }

    vl
}

pub(super) unsafe fn classify_rvv(text: &[u8], tags: &mut [u8]) {
    let mut i = 0;
    while text.len() - i >= 16 {
        if u64::from_ne_bytes(text[i..i + 8].try_into().unwrap()) & 0x8080808080808080 == 0 {
            i += unsafe { ascii_prefix(&text[i..], &mut tags[i..]) };
        } else {
            i += unsafe { mixed(&text[i..], &mut tags[i..]) };
        }
    }
    classify_scalar(&text[i..], &mut tags[i..]);
}
