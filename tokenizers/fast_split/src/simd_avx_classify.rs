//! x86_64 SIMD classify (AVX2 / `vpshufb`), runtime-detected with a scalar fallback.
//!
//! Mirrors `simd_classify.rs` (NEON) but for x86. The DATA is shared — the same `Tables` (built from
//! `S::tables()`) drive both — only the lookup PRIMITIVE differs:
//!   NEON `vqtbl4` is a 64-entry shuffle with out-of-range→0, so the subtract trick chains it to
//!   128/256 entries. AVX2 `vpshufb` is a **16-entry** shuffle **per 128-bit lane** with different
//!   out-of-range semantics (index bit 0x80 → 0; low nibble otherwise), so a 64-entry lookup becomes
//!   the "range trick": add-saturate to fold the index into 0..15 per sub-table + blend. That kernel
//!   is the work; it must be re-derived, not ported.
//!
//! STATUS: scaffold. `dispatch` picks AVX2-vs-scalar correctly; `classify_avx2` is byte-exact via the
//! scalar walk for now so x86 builds/behaves correctly. Filling in the `vpshufb` width kernels
//! (ASCII 128-entry, 2-byte group 256-entry, 3-byte peel 128-entry) is the scoped x86 task — best done
//! on an x86 host where it can be validated against `classify_scalar`.

use super::classify::{classify_scalar, TagScheme};

/// Runtime dispatch: AVX2 if the CPU supports it, else the portable scalar walk.
///
/// Detection via `is_x86_feature_detected!` is cheap enough per-call for now; if it ever shows up in a
/// profile, replace it with memchr's trick — an `AtomicPtr` that starts at a detect-and-choose fn and
/// is overwritten with the chosen implementation after the first call (detect once, then direct-call).
#[inline]
pub fn dispatch<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime AVX2 check above.
        unsafe { classify_avx2::<S>(text, tags) }
    } else {
        classify_scalar::<S>(text, tags)
    }
}

/// AVX2 whole-buffer classify. Same chunk structure as the NEON path (ASCII fast path → 2-byte group
/// → CJK → 3-byte peel → tail → MB fixup), but every table lookup uses `vpshufb` + the range trick.
///
/// TODO(vpshufb kernels): until implemented, delegate to the byte-exact scalar walk. The `Tables` for
/// scheme `S` are available via `S::tables()`; `S::classify_char` covers the tail/astral, exactly as
/// on NEON. Implement + validate (`== classify_scalar::<S>`) on an x86_64 host.
#[target_feature(enable = "avx2")]
unsafe fn classify_avx2<S: TagScheme>(text: &[u8], tags: &mut [u8]) {
    classify_scalar::<S>(text, tags)
}
