//! SIMD helpers for ASCII fast paths.
//!
//! Each public function dispatches at runtime (via `is_x86_feature_detected!`
//! on x86_64; aarch64 always has NEON under the stable target ABI) and falls
//! back to a scalar implementation on other architectures.

/// Lowercase ASCII letters (`A`..=`Z` → `a`..=`z`) in place. Bytes outside that
/// range are left untouched, so it is safe to call on any byte slice — but for
/// best speed it should be guarded by a `is_ascii()` check upstream so the
/// caller can also skip Unicode-aware logic.
#[inline]
pub fn ascii_lower(buf: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { return ascii_lower_avx2(buf) };
        }
        // SSE2 is part of the x86_64 baseline; always available.
        unsafe { return ascii_lower_sse2(buf) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on the stable aarch64 ABI; no runtime check needed.
        unsafe { return ascii_lower_neon(buf) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        ascii_lower_scalar(buf);
    }
}

#[inline(always)]
fn ascii_lower_scalar(buf: &mut [u8]) {
    for b in buf {
        if b.is_ascii_uppercase() {
            *b |= 0x20;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ascii_lower_avx2(buf: &mut [u8]) {
    use std::arch::x86_64::*;

    let a_minus_1 = _mm256_set1_epi8(b'A' as i8 - 1); // 0x40
    let z_plus_1 = _mm256_set1_epi8(b'Z' as i8 + 1);  // 0x5B
    let case_bit = _mm256_set1_epi8(0x20);

    let len = buf.len();
    let mut i = 0;
    while i + 32 <= len {
        let p = buf.as_mut_ptr().add(i) as *mut __m256i;
        let v = _mm256_loadu_si256(p as *const __m256i);
        // Signed compares are correct here because all uppercase ASCII bytes
        // are < 0x80; bytes >= 0x80 appear negative and are excluded from the mask.
        let gt_a = _mm256_cmpgt_epi8(v, a_minus_1);     // v > 0x40
        let lt_z = _mm256_cmpgt_epi8(z_plus_1, v);      // 0x5B > v
        let mask = _mm256_and_si256(gt_a, lt_z);
        let flip = _mm256_and_si256(mask, case_bit);
        let out = _mm256_xor_si256(v, flip);
        _mm256_storeu_si256(p, out);
        i += 32;
    }
    // Scalar tail (also covers buffers shorter than 32 bytes).
    ascii_lower_scalar(&mut buf[i..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn ascii_lower_sse2(buf: &mut [u8]) {
    use std::arch::x86_64::*;

    let a_minus_1 = _mm_set1_epi8(b'A' as i8 - 1);
    let z_plus_1 = _mm_set1_epi8(b'Z' as i8 + 1);
    let case_bit = _mm_set1_epi8(0x20);

    let len = buf.len();
    let mut i = 0;
    while i + 16 <= len {
        let p = buf.as_mut_ptr().add(i) as *mut __m128i;
        let v = _mm_loadu_si128(p as *const __m128i);
        let gt_a = _mm_cmpgt_epi8(v, a_minus_1);
        let lt_z = _mm_cmpgt_epi8(z_plus_1, v);
        let mask = _mm_and_si128(gt_a, lt_z);
        let flip = _mm_and_si128(mask, case_bit);
        let out = _mm_xor_si128(v, flip);
        _mm_storeu_si128(p, out);
        i += 16;
    }
    ascii_lower_scalar(&mut buf[i..]);
}

#[cfg(target_arch = "aarch64")]
unsafe fn ascii_lower_neon(buf: &mut [u8]) {
    use std::arch::aarch64::*;

    let a_minus_1 = vdupq_n_u8(b'A' - 1);
    let z_plus_1 = vdupq_n_u8(b'Z' + 1);
    let case_bit = vdupq_n_u8(0x20);

    let len = buf.len();
    let mut i = 0;
    while i + 16 <= len {
        let p = buf.as_mut_ptr().add(i);
        let v = vld1q_u8(p);
        // Unsigned compares on aarch64 — directly available.
        let gt_a = vcgtq_u8(v, a_minus_1); // v > A-1 → v >= A
        let lt_z = vcltq_u8(v, z_plus_1);  // v < Z+1 → v <= Z
        let mask = vandq_u8(gt_a, lt_z);
        let flip = vandq_u8(mask, case_bit);
        let out = veorq_u8(v, flip);
        vst1q_u8(p, out);
        i += 16;
    }
    ascii_lower_scalar(&mut buf[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_reference(input: &[u8]) -> Vec<u8> {
        let mut out = input.to_vec();
        ascii_lower_scalar(&mut out);
        out
    }

    #[test]
    fn empty() {
        let mut buf: [u8; 0] = [];
        ascii_lower(&mut buf);
    }

    #[test]
    fn matches_scalar_on_random_ascii() {
        // Mix of upper, lower, digits, symbols across many lengths covering
        // sub-block, exact-block, and post-block tails for both 16- and 32-byte
        // SIMD widths.
        let mut data: Vec<u8> = (0..200u32)
            .map(|i| {
                let c = i as u8;
                // Cycle through printable ASCII.
                0x20 + (c % 0x5F)
            })
            .collect();
        let expected = scalar_reference(&data);
        ascii_lower(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn matches_scalar_at_critical_lengths() {
        for len in [0, 1, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 128, 129] {
            let mut data: Vec<u8> = (0..len as u8).map(|i| b'A' + (i % 26)).collect();
            let expected = scalar_reference(&data);
            ascii_lower(&mut data);
            assert_eq!(data, expected, "len={len}");
        }
    }

    #[test]
    fn leaves_high_bytes_untouched() {
        // Ensures SIMD masks correctly exclude bytes >= 0x80 (UTF-8 continuation
        // bytes) — defensive even though the gate is meant to filter these out.
        let seed: Vec<u8> = vec![
            b'A', b'b', 0xC3, 0xA9, b'Z', 0xE2, 0x82, 0xAC, b'Q', 0x80, 0xFF,
        ];
        // Repeat to cross SIMD block boundaries.
        let mut data = seed.repeat(8);
        let expected = scalar_reference(&data);
        ascii_lower(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn idempotent() {
        let mut data = b"Hello, World! THE QUICK BROWN FOX 1234 JUMPS OVER 0 LAZY DOGS.".to_vec();
        ascii_lower(&mut data);
        let once = data.clone();
        ascii_lower(&mut data);
        assert_eq!(data, once);
    }
}
