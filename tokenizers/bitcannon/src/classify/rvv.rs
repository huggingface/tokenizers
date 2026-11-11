//! RVV ASCII classification, with the shared scalar lookup for Unicode.
use super::{ATOM_TABLES, classify_scalar};
use core::arch::asm;

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

pub(super) unsafe fn classify_rvv(text: &[u8], tags: &mut [u8]) {
    let mut i = 0;
    while text.len() - i >= 16 {
        if u64::from_ne_bytes(text[i..i + 8].try_into().unwrap()) & 0x8080808080808080 == 0 {
            // SAFETY: both slices contain the remaining input bytes.
            i += unsafe { ascii_prefix(&text[i..], &mut tags[i..]) };
        } else {
            // Amortize scalar fallback over a block, ending at a codepoint boundary.
            let mut end = (i + 512).min(text.len());
            while end < text.len() && text[end] & 0xc0 == 0x80 {
                end += 1;
            }
            classify_scalar(&text[i..end], &mut tags[i..end]);
            i = end;
        }
    }
    classify_scalar(&text[i..], &mut tags[i..]);
}
