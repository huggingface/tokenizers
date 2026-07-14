//! SIMD fsm kernels — the per-architecture accelerators for the class-family boundary extractor. The
//! scalar core lives in `fsm.rs` (`class_runs_runend`); these are pure perf and BYTE-EXACT with it (the
//! `class_runs_into_matches` sweep in `tests/fsm.rs` proves it on every path). `fsm.rs` dispatches here
//! per arch: aarch64 → NEON, wasm32+simd128 → SIMD128; any other target uses the scalar core, so
//! correctness never depends on a SIMD path being present. x86_64 (SSE/AVX `movemask`) is the natural
//! next port — slot a `class_runs_sse` in the same shape below.
#![allow(dead_code)] // arch-gated: only one target's kernel compiles per build

use crate::classify::{Atom, char_len, in_mask};
use crate::fsm::{Span, emit_class_spans};

/// Class LookUpTable: tag → 0 drop / 1 isolate / 2 keep-A / 3 keep-B; Cont → 0xFF (fill sentinel).
#[inline]
fn class_lut<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>() -> [u8; 16] {
    let mut lut = [3u8; 16];
    let mut t = 0u8;
    while t < 16 {
        lut[t as usize] = if t == Atom::Cont as u8 {
            0xFF
        } else if (DROP >> t) & 1 != 0 {
            0
        } else if (ISOLATE >> t) & 1 != 0 {
            1
        } else if (KEEP_A >> t) & 1 != 0 {
            2
        } else {
            3
        };
        t += 1;
    }
    lut
}

/// Close the open segment at `pos` (emit unless DROP), open a new one of class `cls`.
#[inline(always)]
fn emit(
    out: &mut [Span],
    w: &mut usize,
    seg_start: &mut usize,
    seg_class: &mut u8,
    pos: usize,
    cls: u8,
) {
    if *seg_class != 0 {
        out[*w] = (*seg_start as u32, pos as u32);
        *w += 1;
    }
    *seg_start = pos;
    *seg_class = cls;
}

// ── aarch64 / NEON ──────────────────────────────────────────────────────────────────────────────
/// NEON class-runs boundary-extract: per 16 tags → class via `vqtbl1` (Cont→`0xFF`), fill Cont lanes
/// with the left neighbour's class (≤3 iters = max continuation bytes), then boundary = class-change |
/// isolate lead, restricted to leads → movemask → iterate set bits, emitting one span per non-`DROP`
/// segment. A homogeneous-chunk early-out recovers the run-end bulk-skip for long runs (Digits/CJK).
/// Open segment + carries cross chunks; a scalar tail finishes the < 16-byte remainder.
#[cfg(target_arch = "aarch64")]
pub(crate) fn class_runs_neon<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    use core::arch::aarch64::*;
    // This is a lookup table
    let lut = class_lut::<DROP, ISOLATE, KEEP_A>();
    const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let n = text.len();
    let (mut w, mut i) = (0usize, 0usize);
    let (mut seg_start, mut seg_class) = (0usize, 0u16);
    let mut carry: u8 = 0xFE;
    let mut cls_arr = [0u8; 16];
    // SAFETY: the chunk loop runs only while `i + 16 <= n`, so every `vld1q_u8(tags+i)` reads 16 in-bounds
    // bytes; `vst1q_u8(cls_arr)` writes the 16-byte local array; NEON loads/stores are alignment-free.
    unsafe {
        let lutv = vld1q_u8(lut.as_ptr());
        let powv = vld1q_u8(POW.as_ptr());
        let contv = vdupq_n_u8(0xFF);
        let onev = vdupq_n_u8(1);
        let lonib = vdupq_n_u8(0x0F);
        while i + 16 <= n {
            // Fold the refinement (high nibble) away before the 16-entry LUT: a refined tag (e.g. a
            // lowercase `Letter` = 0x20) indexes out of `vqtbl1`'s 16 lanes, so collapse to the coarse
            // class first — the class-family splits never distinguish refinements. (Cont=15 is untouched.)
            let v = vandq_u8(vld1q_u8(tags.as_ptr().add(i)), lonib);
            let raw = vqtbl1q_u8(lutv, v);
            if seg_class != 1 {
                // we are not at the start, we check the 16 bytes at the same time:
                // raw are the u8 tags. seg class is the running tag or 0
                // contv is the continuation byte. We are just check all are same class or cont ->
                // skip the 16 bytes go to next.
                let ok = vorrq_u8(vceqq_u8(raw, vdupq_n_u8(seg_class)), vceqq_u8(raw, contv));
                if vminvq_u8(ok) == 0xFF {
                    carry = seg_class;
                    i += 16;
                    continue;
                }
            }
            // TODO: I HAVE not read a single thing here.
            let mut cls = raw;
            let mut k = 0;
            while k < 3 {
                let shifted = vextq_u8::<15>(vdupq_n_u8(carry), cls);
                cls = vbslq_u8(vceqq_u8(cls, contv), shifted, cls);
                k += 1;
            }
            let prev = vextq_u8::<15>(vdupq_n_u8(carry), cls);
            let changed = vmvnq_u8(vceqq_u8(cls, prev));
            let is_iso = vceqq_u8(cls, onev);
            let is_lead = vmvnq_u8(vceqq_u8(raw, contv));
            let bnd = vandq_u8(vorrq_u8(changed, vandq_u8(is_iso, is_lead)), is_lead);
            let bits = vandq_u8(bnd, powv);
            let mm =
                (vaddv_u8(vget_low_u8(bits)) as u16) | ((vaddv_u8(vget_high_u8(bits)) as u16) << 8);
            vst1q_u8(cls_arr.as_mut_ptr(), cls);
            let mut m = mm;
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                emit(
                    out,
                    &mut w,
                    &mut seg_start,
                    &mut seg_class,
                    i + j,
                    cls_arr[j],
                );
                m &= m - 1;
            }
            carry = cls_arr[15];
            i += 16;
        }
    }
    emit_class_spans::<DROP, ISOLATE, KEEP_A>(text, tags, out, w, i, seg_start, Some(seg_class))
}

// ── wasm32 / SIMD128 ────────────────────────────────────────────────────────────────────────────

/// SIMD128 class-runs boundary-extract — wasm twin of `class_runs_neon`. `u8x16_swizzle` = `vqtbl1`,
/// `v128_bitselect` = `vbsl`, `u8x16_shuffle` = `vext` (lane shift), `u8x16_bitmask` is the native
/// movemask (no `POW` trick). Same algorithm, byte-exact with the scalar core.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) fn class_runs_wasm<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    use core::arch::wasm32::*;
    let lut = class_lut::<DROP, ISOLATE, KEEP_A>();
    let n = text.len();
    let (mut w, mut i) = (0usize, 0usize);
    let (mut seg_start, mut seg_class) = (0usize, 0u8);
    let mut carry: u8 = 0xFE;
    let mut cls_arr = [0u8; 16];
    let lutv = unsafe { v128_load(lut.as_ptr() as *const v128) };
    let contv = u8x16_splat(0xFF);
    let onev = u8x16_splat(1);
    let lonib = u8x16_splat(0x0F);
    while i + 16 <= n {
        // Fold the refinement nibble away before the 16-entry swizzle LUT (see `class_runs_neon`).
        let v = v128_and(
            unsafe { v128_load(tags.as_ptr().add(i) as *const v128) },
            lonib,
        );
        let raw = u8x16_swizzle(lutv, v);
        if seg_class != 1 {
            let ok = v128_or(u8x16_eq(raw, u8x16_splat(seg_class)), u8x16_eq(raw, contv));
            if u8x16_all_true(ok) {
                carry = seg_class;
                i += 16;
                continue;
            }
        }
        let mut cls = raw;
        for _ in 0..3 {
            // shifted[j] = [carry, cls0..cls14][j] — lane 0 from splat(carry), lanes 1..15 from cls
            let shifted =
                u8x16_shuffle::<0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
                    u8x16_splat(carry),
                    cls,
                );
            cls = v128_bitselect(shifted, cls, u8x16_eq(cls, contv));
        }
        let prev = u8x16_shuffle::<0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
            u8x16_splat(carry),
            cls,
        );
        let changed = v128_not(u8x16_eq(cls, prev));
        let is_iso = u8x16_eq(cls, onev);
        let is_lead = v128_not(u8x16_eq(raw, contv));
        let bnd = v128_and(v128_or(changed, v128_and(is_iso, is_lead)), is_lead);
        let mut mm = u8x16_bitmask(bnd);
        unsafe { v128_store(cls_arr.as_mut_ptr() as *mut v128, cls) };
        while mm != 0 {
            let j = mm.trailing_zeros() as usize;
            emit(
                out,
                &mut w,
                &mut seg_start,
                &mut seg_class,
                i + j,
                cls_arr[j],
            );
            mm &= mm - 1;
        }
        carry = cls_arr[15];
        i += 16;
    }
    tail::<DROP, ISOLATE, KEEP_A>(text, tags, out, w, i, seg_start, seg_class)
}
