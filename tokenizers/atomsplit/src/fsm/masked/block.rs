//! Per-target 64-byte block classifiers for the masked scanners. A block loads 64 consecutive
//! bytes into four SIMD registers once; each method then answers one per-byte predicate for the
//! whole block as a u64 bitmask (bit k = byte k passes). Everything downstream of these masks is
//! platform-independent u64 arithmetic in the scheme modules, so this file is the entire
//! per-target surface.
//!
//! Three targets have a kernel: aarch64 (NEON, baseline), x86_64 (SSE2, baseline) and wasm32
//! with `simd128`. Any other target never reaches this module — the `scan_*_masked` entry
//! points delegate to the scalar scans there.

#![allow(dead_code)] // arch-gated: each build compiles one target's kernel, and schemes land
// one by one, so some predicates are unused until their scheme arrives.

// ── aarch64 / NEON ──────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::Block;

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    /// 64 bytes in four NEON registers. `tag` methods fold the refinement nibble away first
    /// (`& 0x0F`), `full`/byte methods compare the raw byte.
    pub(crate) struct Block {
        v: [uint8x16_t; 4],
    }

    /// simdjson's arm64 movemask: 4 mask vectors (64 lanes of 0x00/0xFF) to one u64, bit i =
    /// lane i. The 4-`addp` reduction is pinned as asm (via gigatoken's `mask.rs`, MIT): written
    /// with `vpaddq_u8`, LLVM rewrites the pairwise adds into uzp1/uzp2/orr triples and the call
    /// grows from 9 to 17 vector ops.
    #[inline(always)]
    unsafe fn movemask64(v0: uint8x16_t, v1: uint8x16_t, v2: uint8x16_t, v3: uint8x16_t) -> u64 {
        // SAFETY: pure NEON register arithmetic, no memory access beyond the 16-byte constant.
        unsafe {
            const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let w = vld1q_u8(W.as_ptr());
            let mut a0 = vandq_u8(v0, w);
            let a1 = vandq_u8(v1, w);
            let a2 = vandq_u8(v2, w);
            let a3 = vandq_u8(v3, w);
            core::arch::asm!(
                "addp {a0:v}.16b, {a0:v}.16b, {a1:v}.16b",
                "addp {a2:v}.16b, {a2:v}.16b, {a3:v}.16b",
                "addp {a0:v}.16b, {a0:v}.16b, {a2:v}.16b",
                "addp {a0:v}.16b, {a0:v}.16b, {a0:v}.16b",
                a0 = inout(vreg) a0,
                a1 = in(vreg) a1,
                a2 = inout(vreg) a2 => _,
                a3 = in(vreg) a3,
                options(pure, nomem, nostack, preserves_flags),
            );
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a0))
        }
    }

    impl Block {
        /// Load `bytes[at..at + 64]`.
        ///
        /// # Safety
        ///
        /// `at + 64 <= bytes.len()`. NEON loads are alignment-free.
        #[inline(always)]
        pub(crate) unsafe fn load(bytes: &[u8], at: usize) -> Self {
            debug_assert!(at + 64 <= bytes.len());
            // SAFETY: the fn contract puts all four 16-byte loads in bounds.
            unsafe {
                let p = bytes.as_ptr().add(at);
                Self {
                    v: [
                        vld1q_u8(p),
                        vld1q_u8(p.add(16)),
                        vld1q_u8(p.add(32)),
                        vld1q_u8(p.add(48)),
                    ],
                }
            }
        }

        #[inline(always)]
        fn mask(&self, f: impl Fn(uint8x16_t) -> uint8x16_t) -> u64 {
            // SAFETY: register arithmetic only.
            unsafe { movemask64(f(self.v[0]), f(self.v[1]), f(self.v[2]), f(self.v[3])) }
        }

        #[inline(always)]
        fn any(&self, f: impl Fn(uint8x16_t) -> uint8x16_t) -> bool {
            // SAFETY: register arithmetic only.
            unsafe {
                let o = vorrq_u8(
                    vorrq_u8(f(self.v[0]), f(self.v[1])),
                    vorrq_u8(f(self.v[2]), f(self.v[3])),
                );
                vmaxvq_u8(o) != 0
            }
        }

        /// Bytes whose low nibble equals `k`.
        #[inline(always)]
        pub(crate) fn eq_tag(&self, k: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe { vceqq_u8(vandq_u8(v, vdupq_n_u8(0x0F)), vdupq_n_u8(k)) })
        }

        /// Bytes in `lo..=lo + span` (the raw byte — text blocks).
        #[inline(always)]
        pub(crate) fn range_full(&self, lo: u8, span: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe { vcleq_u8(vsubq_u8(v, vdupq_n_u8(lo)), vdupq_n_u8(span)) })
        }

        /// Bytes whose low nibble is in `lo..=lo + span`.
        #[inline(always)]
        pub(crate) fn range_tag(&self, lo: u8, span: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                vcleq_u8(
                    vsubq_u8(vandq_u8(v, vdupq_n_u8(0x0F)), vdupq_n_u8(lo)),
                    vdupq_n_u8(span),
                )
            })
        }

        /// Bytes equal to `k` (the raw byte — refined tags, or text bytes).
        #[inline(always)]
        pub(crate) fn eq_full(&self, k: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe { vceqq_u8(v, vdupq_n_u8(k)) })
        }

        /// Any byte with low nibble in `lo..=lo + span`? (Cheaper than `range_tag` when only
        /// presence matters — no movemask.)
        #[inline(always)]
        pub(crate) fn any_range_tag(&self, lo: u8, span: u8) -> bool {
            // SAFETY: register arithmetic only.
            self.any(|v| unsafe {
                vcleq_u8(
                    vsubq_u8(vandq_u8(v, vdupq_n_u8(0x0F)), vdupq_n_u8(lo)),
                    vdupq_n_u8(span),
                )
            })
        }

        /// Any byte equal to `k`?
        #[inline(always)]
        pub(crate) fn any_eq_full(&self, k: u8) -> bool {
            // SAFETY: register arithmetic only.
            self.any(|v| unsafe { vceqq_u8(v, vdupq_n_u8(k)) })
        }

        /// ASCII letters `[A-Za-z]` (text blocks).
        #[inline(always)]
        pub(crate) fn ascii_alpha(&self) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                vcleq_u8(
                    vsubq_u8(vorrq_u8(v, vdupq_n_u8(0x20)), vdupq_n_u8(b'a')),
                    vdupq_n_u8(25),
                )
            })
        }

        /// ASCII punctuation (the four `is_ascii_punctuation` ranges, text blocks).
        #[inline(always)]
        pub(crate) fn ascii_punct(&self) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                let r = |v: uint8x16_t, lo: u8, span: u8| {
                    vcleq_u8(vsubq_u8(v, vdupq_n_u8(lo)), vdupq_n_u8(span))
                };
                vorrq_u8(
                    vorrq_u8(r(v, 0x21, 0x0E), r(v, 0x3A, 0x06)),
                    vorrq_u8(r(v, 0x5B, 0x05), r(v, 0x7B, 0x03)),
                )
            })
        }
    }
}

// ── x86_64 / SSE2 ──────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
pub(crate) use sse2::Block;

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;

    /// 64 bytes in four SSE2 registers; `pmovmskb` is the native movemask, 16 bits per register.
    /// SSE2 is baseline on x86_64, so no runtime detection is needed. (SSE2 has no unsigned
    /// byte compare: `x <= span` is done as `max(x, span) == span`.)
    pub(crate) struct Block {
        v: [__m128i; 4],
    }

    #[inline(always)]
    fn movemask64(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> u64 {
        // SAFETY: register arithmetic only; SSE2 is baseline on x86_64.
        unsafe {
            (_mm_movemask_epi8(v0) as u16 as u64)
                | ((_mm_movemask_epi8(v1) as u16 as u64) << 16)
                | ((_mm_movemask_epi8(v2) as u16 as u64) << 32)
                | ((_mm_movemask_epi8(v3) as u16 as u64) << 48)
        }
    }

    #[inline(always)]
    fn le(x: __m128i, span: u8) -> __m128i {
        // SAFETY: register arithmetic only.
        unsafe {
            let s = _mm_set1_epi8(span as i8);
            _mm_cmpeq_epi8(_mm_max_epu8(x, s), s)
        }
    }

    impl Block {
        /// Load `bytes[at..at + 64]`.
        ///
        /// # Safety
        ///
        /// `at + 64 <= bytes.len()`. Unaligned loads (`loadu`).
        #[inline(always)]
        pub(crate) unsafe fn load(bytes: &[u8], at: usize) -> Self {
            debug_assert!(at + 64 <= bytes.len());
            // SAFETY: the fn contract puts all four 16-byte loads in bounds.
            unsafe {
                let p = bytes.as_ptr().add(at);
                Self {
                    v: [
                        _mm_loadu_si128(p.cast()),
                        _mm_loadu_si128(p.add(16).cast()),
                        _mm_loadu_si128(p.add(32).cast()),
                        _mm_loadu_si128(p.add(48).cast()),
                    ],
                }
            }
        }

        #[inline(always)]
        fn mask(&self, f: impl Fn(__m128i) -> __m128i) -> u64 {
            movemask64(f(self.v[0]), f(self.v[1]), f(self.v[2]), f(self.v[3]))
        }

        #[inline(always)]
        pub(crate) fn eq_tag(&self, k: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                _mm_cmpeq_epi8(
                    _mm_and_si128(v, _mm_set1_epi8(0x0F)),
                    _mm_set1_epi8(k as i8),
                )
            })
        }

        #[inline(always)]
        pub(crate) fn range_tag(&self, lo: u8, span: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                le(
                    _mm_sub_epi8(
                        _mm_and_si128(v, _mm_set1_epi8(0x0F)),
                        _mm_set1_epi8(lo as i8),
                    ),
                    span,
                )
            })
        }

        #[inline(always)]
        pub(crate) fn range_full(&self, lo: u8, span: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe { le(_mm_sub_epi8(v, _mm_set1_epi8(lo as i8)), span) })
        }

        #[inline(always)]
        pub(crate) fn eq_full(&self, k: u8) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe { _mm_cmpeq_epi8(v, _mm_set1_epi8(k as i8)) })
        }

        #[inline(always)]
        pub(crate) fn any_range_tag(&self, lo: u8, span: u8) -> bool {
            self.range_tag(lo, span) != 0
        }

        #[inline(always)]
        pub(crate) fn any_eq_full(&self, k: u8) -> bool {
            self.eq_full(k) != 0
        }

        #[inline(always)]
        pub(crate) fn ascii_alpha(&self) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                le(
                    _mm_sub_epi8(
                        _mm_or_si128(v, _mm_set1_epi8(0x20)),
                        _mm_set1_epi8(b'a' as i8),
                    ),
                    25,
                )
            })
        }

        #[inline(always)]
        pub(crate) fn ascii_punct(&self) -> u64 {
            // SAFETY: register arithmetic only.
            self.mask(|v| unsafe {
                let r = |v: __m128i, lo: u8, span: u8| {
                    le(_mm_sub_epi8(v, _mm_set1_epi8(lo as i8)), span)
                };
                _mm_or_si128(
                    _mm_or_si128(r(v, 0x21, 0x0E), r(v, 0x3A, 0x06)),
                    _mm_or_si128(r(v, 0x5B, 0x05), r(v, 0x7B, 0x03)),
                )
            })
        }
    }
}

// ── wasm32 / SIMD128 ────────────────────────────────────────────────────────────────────────────
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::Block;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::*;

    /// 64 bytes in four v128 registers; `u8x16_bitmask` is the native movemask.
    pub(crate) struct Block {
        v: [v128; 4],
    }

    #[inline(always)]
    fn movemask64(v0: v128, v1: v128, v2: v128, v3: v128) -> u64 {
        (u8x16_bitmask(v0) as u64)
            | ((u8x16_bitmask(v1) as u64) << 16)
            | ((u8x16_bitmask(v2) as u64) << 32)
            | ((u8x16_bitmask(v3) as u64) << 48)
    }

    impl Block {
        /// Load `bytes[at..at + 64]`.
        ///
        /// # Safety
        ///
        /// `at + 64 <= bytes.len()`. `v128_load` is alignment-free.
        #[inline(always)]
        pub(crate) unsafe fn load(bytes: &[u8], at: usize) -> Self {
            debug_assert!(at + 64 <= bytes.len());
            // SAFETY: the fn contract puts all four 16-byte loads in bounds.
            unsafe {
                let p = bytes.as_ptr().add(at);
                Self {
                    v: [
                        v128_load(p.cast()),
                        v128_load(p.add(16).cast()),
                        v128_load(p.add(32).cast()),
                        v128_load(p.add(48).cast()),
                    ],
                }
            }
        }

        #[inline(always)]
        fn mask(&self, f: impl Fn(v128) -> v128) -> u64 {
            movemask64(f(self.v[0]), f(self.v[1]), f(self.v[2]), f(self.v[3]))
        }

        #[inline(always)]
        pub(crate) fn eq_tag(&self, k: u8) -> u64 {
            self.mask(|v| u8x16_eq(v128_and(v, u8x16_splat(0x0F)), u8x16_splat(k)))
        }

        #[inline(always)]
        pub(crate) fn range_tag(&self, lo: u8, span: u8) -> u64 {
            self.mask(|v| {
                u8x16_le(
                    u8x16_sub(v128_and(v, u8x16_splat(0x0F)), u8x16_splat(lo)),
                    u8x16_splat(span),
                )
            })
        }

        #[inline(always)]
        pub(crate) fn range_full(&self, lo: u8, span: u8) -> u64 {
            self.mask(|v| u8x16_le(u8x16_sub(v, u8x16_splat(lo)), u8x16_splat(span)))
        }

        #[inline(always)]
        pub(crate) fn eq_full(&self, k: u8) -> u64 {
            self.mask(|v| u8x16_eq(v, u8x16_splat(k)))
        }

        #[inline(always)]
        pub(crate) fn any_range_tag(&self, lo: u8, span: u8) -> bool {
            self.range_tag(lo, span) != 0
        }

        #[inline(always)]
        pub(crate) fn any_eq_full(&self, k: u8) -> bool {
            self.eq_full(k) != 0
        }

        #[inline(always)]
        pub(crate) fn ascii_alpha(&self) -> u64 {
            self.mask(|v| {
                u8x16_le(
                    u8x16_sub(v128_or(v, u8x16_splat(0x20)), u8x16_splat(b'a')),
                    u8x16_splat(25),
                )
            })
        }

        #[inline(always)]
        pub(crate) fn ascii_punct(&self) -> u64 {
            self.mask(|v| {
                let r = |v: v128, lo: u8, span: u8| {
                    u8x16_le(u8x16_sub(v, u8x16_splat(lo)), u8x16_splat(span))
                };
                v128_or(
                    v128_or(r(v, 0x21, 0x0E), r(v, 0x3A, 0x06)),
                    v128_or(r(v, 0x5B, 0x05), r(v, 0x7B, 0x03)),
                )
            })
        }
    }
}
