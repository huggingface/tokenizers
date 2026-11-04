//! RVV byte matching for the literal splitter's 64-byte blocks.
use core::arch::asm;
use std::sync::OnceLock;

/// Linux HWCAP exposes the standard V extension as bit ('V' - 'A').
#[inline]
pub(crate) fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        unsafe extern "C" {
            fn getauxval(kind: usize) -> usize;
        }
        // AT_HWCAP is process-wide; this does not assume a particular VLEN.
        unsafe { getauxval(16) & (1 << (b'V' - b'A')) != 0 }
    })
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
