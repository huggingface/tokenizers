// NEON nibble-classifier: build two 16-byte tables for an arbitrary byte set, classify 16 bytes/instr,
// and find the first byte NOT in the set (run-end). Scalar fallback off aarch64 / when >8 high nibbles.
pub struct Cls { pub lo: [u8; 16], pub hi: [u8; 16], pub cand: [bool; 256], pub ok: bool }

pub fn build_cls(member: impl Fn(u8) -> bool) -> Cls {
    let mut cand = [false; 256];
    for b in 0..256 { cand[b] = member(b as u8); }
    let (mut lo, mut hi) = ([0u8; 16], [0u8; 16]);
    let (mut next, mut ok) = (0u32, true);
    for h in 0..16usize {
        if (0..16).any(|l| cand[(h << 4) | l]) {
            if next >= 8 { ok = false; break; }
            hi[h] = 1u8 << next; next += 1;
        }
    }
    if ok { for h in 0..16usize { for l in 0..16usize { if cand[(h << 4) | l] { lo[l] |= hi[h]; } } } }
    Cls { lo, hi, cand, ok }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_first_not(t: &[u8], from: usize, end: usize, lo: &[u8; 16], hi: &[u8; 16]) -> usize {
    use core::arch::aarch64::*;
    let lov = vld1q_u8(lo.as_ptr());
    let hiv = vld1q_u8(hi.as_ptr());
    let m0f = vdupq_n_u8(0x0f);
    let mut i = from;
    while i + 16 <= end {
        let v = vld1q_u8(t.as_ptr().add(i));
        let low = vandq_u8(v, m0f);
        let high = vshrq_n_u8(v, 4);
        let lm = vqtbl1q_u8(lov, low);
        let hm = vqtbl1q_u8(hiv, high);
        let mm = vandq_u8(lm, hm);                 // nonzero lane == in set
        let notin = vceqzq_u8(mm);                 // 0xFF where NOT in set
        let packed = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(notin), 4)), 0);
        if packed != 0 { return i + (packed.trailing_zeros() as usize >> 2); }
        i += 16;
    }
    i
}

#[inline]
pub fn first_not(t: &[u8], from: usize, end: usize, cls: &Cls) -> usize {
    let mut i = from;
    #[cfg(target_arch = "aarch64")]
    {
        if cls.ok && end >= from + 16 { unsafe { i = neon_first_not(t, from, end, &cls.lo, &cls.hi); } }
    }
    while i < end { if !cls.cand[t[i] as usize] { return i; } i += 1; }
    end
}
