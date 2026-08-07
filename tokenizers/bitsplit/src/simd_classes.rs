//! SIMD fsm kernels — the per-architecture accelerators for the class-family boundary extractor. The
//! scalar core lives in `fsm.rs` (`emit_class_spans`); these are pure perf and BYTE-EXACT with it (the
//! `class_runs_into_matches` sweep in `tests/fsm.rs` proves it on every path). `fsm.rs` dispatches here
//! per arch: aarch64 → NEON, wasm32+simd128 → SIMD128; any other target uses the scalar core, so
//! correctness never depends on a SIMD path being present. x86_64 (SSE/AVX `movemask`) is the natural
//! next port — slot a `class_runs_sse` in the same shape below.
#![allow(dead_code, unused_imports)] // arch-gated: only one target's kernel compiles per build, so on
// the non-selected arch some items/imports (char_len, emit_class_spans, ...) are legitimately unused

use crate::classify::{Atom, char_len};
use crate::Span;
use crate::classes::emit_class_spans;

/// Class LookUpTable: tag → 0 drop / 1 isolate / 2 keep-A / 3 keep-B; Cont → 0xFF (fill sentinel).
/// This lookup table is built per parameter DROP, ISOLATE and KEEP_A. These as
/// [`crate::classify::mask`] u16 bitmap masks. The LUT is indexed with a low nibble u4 (0..15) and
/// gives the behavior {0, 1, 2, 3} for the class.
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
        out[*w] = Span {
            start: *seg_start as u32,
            end: pos as u32,
        };
        *w += 1;
    }
    *seg_start = pos;
    *seg_class = cls;
}

/// SIMD→scalar handoff: finish the `< 16`-byte remainder from `i` via the scalar core. The kernels
/// carry the open segment as a `class_lut` *code* (0=drop, 1=isolate, 2=keep-A, 3=keep-B/other; and the
/// initial 0 = "leading drop run", a no-op when DROP is empty), but [`emit_class_spans`] wants that
/// segment's tag *mask* — translate, then defer. An open isolate is a single pending char, not a run, so
/// close it here before scanning the rest fresh (`emit_class_spans`'s open-segment path assumes a run).
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline]
fn tail<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
    w: usize,
    i: usize,
    seg_start: usize,
    seg_class: u8,
) -> usize {
    if seg_class == 1 {
        // open isolate = one pending char; emit it, then scan the remainder with no open segment.
        let e = seg_start + char_len(text[seg_start]);
        out[w] = Span {
            start: seg_start as u32,
            end: e as u32,
        };
        return emit_class_spans::<DROP, ISOLATE, KEEP_A>(text, tags, out, w + 1, e, 0, None);
    }
    // seg_mask of the last bytes before the tail converted to a full u16 mask. The simd path uses
    // a u8.
    let seg_mask = match seg_class {
        0 => DROP,
        2 => KEEP_A,
        _ => !(DROP | ISOLATE | KEEP_A), // 3 = keep-B / other
    };
    emit_class_spans::<DROP, ISOLATE, KEEP_A>(text, tags, out, w, i, seg_start, Some(seg_mask))
}

// ── aarch64 / NEON ──────────────────────────────────────────────────────────────────────────────
//  This is the same as emit_class_spans but using SIMD. We emit spans at class boundaries.
//  A `tag` is the coarse Atom (L=letter W=whitespace P=punct C=UTF-8 cont). The `class` is
//  class_lut[tag & 0x0F] for THIS pretokenizer's masks:  0=drop 1=isolate 2=keep-A 3=keep-B
//  (Cont → 0xFF, shown ·, a sentinel filled from the left neighbour in STEP 3).
//  In this worked example the masks give  L→2 (keep-A),  W→0 (drop),  P→3 (keep-B):
//
//            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   ← lane
// byte       H  i  ␣ E4 B8 96  ␣  w  o  r  l  d  ␣  !  .  x
//            └Hi┘ ws └─ 世 ──┘ ws └─ world ──┘ ws └!.┘ x
// tag        L  L  W  L  C  C  W  L  L  L  L  L  W  P  P  L
//
// ┌─ STEP 1  load 16 tags into one NEON register ────────────────────────────┐
//
// ┌─ STEP 2  tag → class   (vqtbl1q_u8 = one 16-wide LUT; L→2 W→0 P→3, Cont→·) ┐
// class      2  2  0  2  ·  ·  0  2  2  2  2  2  0  3  3  2
//
// ┌─ STEP 3  fill Cont (·) from the left neighbour  (≤3 vext+vbsl; 世's · ← 2) ─┐
// class'     2  2  0  2  2  2  0  2  2  2  2  2  0  3  3  2
//                     └──┘  every cont lane now carries its char's class (2)
//
// ┌─ STEP 4  prev = class' shifted right 1 lane   (lane0 ← carry = 0) ────────┐
// prev       0  2  2  0  2  2  2  0  2  2  2  2  2  0  3  3
//            ↑carry (a class 0..3) from previous chunk's last lane
//
// ┌─ STEP 5  boundary = (class' ≠ prev)  [isolate (class 1) lanes also forced] ┐
// bound      1  .  1  1  .  .  1  1  .  .  .  .  1  1  .  1
//
// ┌─ STEP 6  movemask → 16-bit int;  set bits = {0,2,3,6,7,12,13,15} ─────────┐
//            iterate set bits: each is a token start → span to the next start
// Open segment + carries cross chunks; a scalar tail finishes the < 16-byte remainder.
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
    let (mut seg_start, mut seg_class) = (0usize, 0u8);
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
            // Lookup for each low nibble (the coarse class like Letter) what the input parameter
            // require doing. raw is in 0..3
            let raw = vqtbl1q_u8(lutv, v);
            if seg_class != 1 {
                // Fast-skip: an isolate segment (class 1) is a single char, never a run, so only a
                // run-forming open segment can extend. Check every lane already has that open class
                // (the OR lets cont lanes 0xFF pass — they inherit their lead's class in STEP 3).
                let ok = vorrq_u8(vceqq_u8(raw, vdupq_n_u8(seg_class)), vceqq_u8(raw, contv));
                if vminvq_u8(ok) == 0xFF {
                    // no lane differs → no boundary in this chunk; extend the open segment, advance.
                    carry = seg_class;
                    i += 16;
                    continue;
                }
            }
            let mut cls = raw;
            let mut k = 0;
            // ┌─ STEP 3  fill Cont (·=0xFF) from the left neighbour  (≤3 vext+vbsl) ───────┐
            //   class = class_lut[tag & 0x0F] ∈ {0 drop, 1 isolate, 2 keep-A, 3 keep-B};  Cont → ·
            //   (this example's masks:  L→2  W→0  P→3)
            // tag        L  L  W  L  C  C  W  L  L  L  L  L  W  P  P  L
            // class      2  2  0  2  ·  ·  0  2  2  2  2  2  0  3  3  2   ← raw LUT (Cont = ·)
            // class'     2  2  0  2  2  2  0  2  2  2  2  2  0  3  3  2   ← 世's two · ← left's 2
            //                     └──┘  every cont lane now carries its char's class
            while k < 3 {
                // vext::<N>  does: cat(carry[15..], cls[..15])
                // creating: [carry, cls[0], ..., cls[14]].
                let shifted = vextq_u8::<15>(vdupq_n_u8(carry), cls);
                // we fill the cls vector at the index of continuatuion bytes with
                // the values of shifted. this allows us to fill continuations bytes.
                cls = vbslq_u8(vceqq_u8(cls, contv), shifted, cls);
                k += 1;
            }
            // ┌─ STEP 4  prev = class' shifted right 1 lane   (lane0 ← carry = 0) ────────┐
            // class'     2  2  0  2  2  2  0  2  2  2  2  2  0  3  3  3
            // prev       0  2  2  0  2  2  2  0  2  2  2  2  2  0  3  3
            //            ↑carry (a class 0..3) from the previous chunk's last lane
            let prev = vextq_u8::<15>(vdupq_n_u8(carry), cls);
            // ┌─ STEP 5  boundary = (class' ≠ prev)  [isolate (class 1) lanes also forced] ┐
            // bound      1  .  1  1  .  .  1  1  .  .  .  .  1  1  .  1
            let changed = vmvnq_u8(vceqq_u8(cls, prev));
            let is_iso = vceqq_u8(cls, onev);
            let is_lead = vmvnq_u8(vceqq_u8(raw, contv));
            let bnd = vandq_u8(vorrq_u8(changed, vandq_u8(is_iso, is_lead)), is_lead);
            let bits = vandq_u8(bnd, powv);
            // ┌─ STEP 6  movemask → 16-bit int;  set bits = {0,2,3,6,7,12,13,15} ─────────┐
            //            iterate set bits: each is a token start → span to the next start
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
    tail::<DROP, ISOLATE, KEEP_A>(text, tags, out, w, i, seg_start, seg_class)
}

// ── wasm32 / SIMD128 ────────────────────────────────────────────────────────────────────────────

/// ⚠️ AI-GENERATED, NOT NECESSARILY REVIEWED — ported from the hand-crafted `class_runs_neon` above
/// (the source of truth); trusted only insofar as the byte-exactness tests pass.
///
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
