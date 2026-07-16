//! The complete scalar normalization core — portable to every arch and the byte-exactness oracle
//! for `simd_norm` (whose kernels are pure prefix processors: they stop and these loops finish).
//! See `lib.rs` for the tag encoding and the layered-skip design.
use crate::tables::*;
use std::borrow::Cow;

const HANGUL: std::ops::RangeInclusive<u32> = 0xAC00..=0xD7A3;
// per-form lead-mask bit: decompose uses bits 0/1, compose bits 2/3.
const fn d_bit<const K: bool>() -> u8 {
    if K { 2 } else { 1 }
}
const fn c_bit<const K: bool>() -> u8 {
    if K { 8 } else { 4 }
}

/// Tag of any codepoint (BMP: direct index; astral: RLE binary search — rare by construction).
#[inline]
fn tag(cp: u32) -> u8 {
    if cp < 0x10000 {
        TAG[cp as usize]
    } else {
        match ASTRAL.binary_search_by(|&(s, _)| s.cmp(&cp)) {
            Ok(k) => ASTRAL[k].1,
            Err(k) => ASTRAL[k - 1].1, // RLE: value of the run containing cp
        }
    }
}

/// Decomposition changes the char under this form?
#[inline]
fn d_breaks<const K: bool>(t: u8) -> bool {
    t >= 0x7D || (K && matches!(t & 0x3F, 0x3C | 0x3D))
}
/// ccc rank for order checks (0 = starter). `0x3D` (odd compat mark) is handled by its callers.
#[inline]
fn rank(t: u8) -> u8 {
    let r = t & 0x3F;
    if r >= 0x3C { 0 } else { r }
}
/// Composition must recompose at this char?
#[inline]
fn c_relevant<const K: bool>(t: u8) -> bool {
    (t & 0x40 != 0 && t < 0x7D) || t == 0x7D || t & 0x3F == 0x3D || (K && t & 0x3F == 0x3C)
}

/// Decode the UTF-8 char at `i` → (codepoint, width). `bytes` is valid UTF-8 at a boundary.
#[inline]
pub(crate) fn decode_cp(bytes: &[u8], i: usize) -> (u32, usize) {
    let b0 = bytes[i];
    if b0 < 0x80 {
        return (b0 as u32, 1);
    }
    // SAFETY: valid UTF-8 ⇒ the continuation bytes exist.
    unsafe {
        let c = |k: usize| (*bytes.get_unchecked(i + k) & 0x3F) as u32;
        if b0 < 0xE0 {
            ((((b0 & 0x1F) as u32) << 6) | c(1), 2)
        } else if b0 < 0xF0 {
            ((((b0 & 0x0F) as u32) << 12) | (c(1) << 6) | c(2), 3)
        } else {
            (
                (((b0 & 0x07) as u32) << 18) | (c(1) << 12) | (c(2) << 6) | c(3),
                4,
            )
        }
    }
}

#[inline]
pub(crate) fn bmp_set(cp: u16) -> bool {
    BMP_SET[(cp >> 6) as usize] >> (cp & 63) & 1 != 0
}

// ── the three skip kernels: scalar loops, SIMD-prefixed on aarch64 ────────────────────────────────

/// The form's 64-entry lead mask (0xFF = some char with this lead needs work under the form).
#[inline]
fn lead_mask<const FB: u8>() -> &'static [u8; 64] {
    match FB {
        1 => &LEAD_NFD,
        2 => &LEAD_NFKD,
        4 => &LEAD_NFC,
        _ => &LEAD_NFKC,
    }
}

/// Advance over bytes that provably start nothing the form cares about: ASCII, continuations, and
/// chars whose LEAD is clean for form-bit `FB`. One `vqtbl4` per 16 bytes; `STORE` write-through
/// rides the caller's `+16` capacity slack. Returns the first suspect-lead position (a char boundary).
#[inline]
fn skip_clean<const FB: u8, const STORE: bool, const SIMD: bool>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
) -> usize {
    let n = bytes.len();
    let mask = lead_mask::<FB>();
    #[cfg(target_arch = "aarch64")]
    if SIMD {
        i = crate::simd_norm::skip_clean::<STORE>(bytes, i, out, mask);
    }
    // scalar tail (and the portable path)
    while i < n {
        let b = bytes[i];
        if b >= 0xC0 && mask[(b - 0xC0) as usize] != 0 {
            return i;
        }
        if STORE {
            // raw byte push (whole chars arrive byte-by-byte; suspect stops are boundaries)
            // SAFETY: clean spans are copied verbatim, so the String stays valid UTF-8.
            unsafe { out.as_mut_vec().push(b) };
        }
        i += 1;
    }
    i
}

// ── layer 1: per-width suspect kernels ────────────────────────────────────────────────────────────

/// 4-byte (astral) chars: relevant leads are only `F0` with a handful of `b1` values (`ASTRAL_B1`).
/// Returns the tag (0 = clean) — pure scalar, astral suspects are rare by construction.
#[inline]
fn astral_tag(bytes: &[u8], i: usize) -> (u8, u32) {
    if bytes[i] != 0xF0 || ASTRAL_B1 >> (bytes[i + 1] & 0x3F) & 1 == 0 {
        return (0, 0);
    }
    let (cp, _) = decode_cp(bytes, i);
    (tag(cp), cp)
}

/// Fused ASCII + 2-byte skip for word-structured 2-byte scripts (Hebrew/Cyrillic/Greek/Arabic —
/// median non-ASCII run is ~5 chars, so per-word kernel round-trips would dominate): 16 BYTES per
/// chunk, byte-classed as ascii | continuation | 2-byte lead, with the char's union bit probed
/// IN-REGISTER at its continuation lane — `idx = (prev & 0x1F) << 3 | (cur & 0x3F) >> 3` keeps the
/// whole 2048-bit bitmap lookup in 8-bit lanes (a 256-byte `vqtbl` table = `BMP_SET[..32]`).
/// Returns `(pos, cp)`: set char (decoded) or a non-(ascii|2-byte) boundary with `cp == 0`.
#[inline]
fn skip2_ascii<const STORE: bool, const SIMD: bool>(bytes: &[u8], mut i: usize, out: &mut String) -> (usize, u32) {
    let n = bytes.len();
    #[cfg(target_arch = "aarch64")]
    if SIMD {
        i = crate::simd_norm::skip2_ascii::<STORE>(bytes, i, out);
    }
    // scalar tail (and the portable path)
    while i < n {
        let b = bytes[i];
        if b < 0x80 {
            if STORE {
                // SAFETY: verbatim ASCII byte.
                unsafe { out.as_mut_vec().push(b) };
            }
            i += 1;
            continue;
        }
        if !(0xC2..=0xDF).contains(&b) {
            return (i, 0);
        }
        let (cp, _) = decode_cp(bytes, i);
        if bmp_set(cp as u16) {
            return (i, cp);
        }
        if STORE {
            // SAFETY: verbatim whole char.
            unsafe { raw_extend(out, bytes.as_ptr().add(i), 2, 2) };
        }
        i += 2;
    }
    (n, 0)
}

/// Scan a uniform 3-byte run from `i`, probing the union bitmap per decoded char. Returns
/// `(pos, cp)`: `cp != 0` = a bitmap-set char (already decoded); `cp == 0` = the run ended (width
/// change / tail) at `pos`. `STORE` writes verified bytes through (`vstNq` of the loaded registers).
#[inline]
fn skip3<const STORE: bool, const SIMD: bool>(bytes: &[u8], mut i: usize, out: &mut String) -> (usize, u32) {
    let n = bytes.len();
    #[cfg(target_arch = "aarch64")]
    if SIMD {
        i = crate::simd_norm::skip3::<STORE>(bytes, i, out);
    }
    // scalar tail (and the portable path)
    while i < n {
        let b = bytes[i];
        let ok = (0xE0..=0xEF).contains(&b);
        if !ok {
            return (i, 0);
        }
        let (cp, _) = decode_cp(bytes, i);
        if bmp_set(cp as u16) {
            return (i, cp);
        }
        if STORE {
            out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[i..i + 3]) });
        }
        i += 3;
    }
    (i, 0)
}

/// The whole scan, centralized: layer-0 clean-byte skip, decode-FIRST at suspect leads (dense
/// suspects — Hebrew marks, French accents — never pay a kernel round-trip), and a width kernel to
/// skim runs only when consecutive chars are bitmap-clear under a dirty lead. Returns the next
/// union-set char `(pos, cp)` or `(len, 0)`. `STORE` writes everything skipped through to `out`.
#[inline]
fn next_suspect<const FB: u8, const STORE: bool, const SIMD: bool>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
) -> (usize, u32) {
    let n = bytes.len();
    loop {
        if i >= n {
            return (n, 0);
        }
        let b = bytes[i];
        if b < 0xC0 || lead_mask::<FB>()[(b - 0xC0) as usize] == 0 {
            i = skip_clean::<FB, STORE, SIMD>(bytes, i, out);
            if i >= n {
                return (n, 0);
            }
        }
        let b = bytes[i];
        let w = if b < 0xE0 {
            2
        } else if b < 0xF0 {
            3
        } else {
            4
        };
        if w == 4 {
            let (t, cp) = astral_tag(bytes, i);
            if t != 0 {
                return (i, cp);
            }
            if STORE {
                // SAFETY: exact in-bounds copy; capacity reserved.
                unsafe { raw_extend(out, bytes.as_ptr().add(i), 4, 4) };
            }
            i += 4;
            continue;
        }
        if w == 2 {
            // decode-first: dense suspects (French é, Greek ά — every hit is set) never pay the
            // kernel entry; a short clear streak escalates to the fused kernel for real words
            let mut streak = 0u32;
            loop {
                if i + 2 > n || !(0xC2..=0xDF).contains(&bytes[i]) {
                    break;
                }
                let cp = (((bytes[i] & 0x1F) as u32) << 6) | (bytes[i + 1] & 0x3F) as u32;
                if bmp_set(cp as u16) {
                    return (i, cp);
                }
                if STORE {
                    // SAFETY: verbatim whole char.
                    unsafe { raw_extend(out, bytes.as_ptr().add(i), 2, 2) };
                }
                i += 2;
                streak += 1;
                if streak == 4 {
                    let (pos, cp2) = skip2_ascii::<STORE, SIMD>(bytes, i, out);
                    if cp2 != 0 || pos >= n {
                        return (pos, cp2);
                    }
                    i = pos;
                    break;
                }
            }
            continue;
        }
        // 3-byte suspects: scalar-first (runs are short in marked scripts); escalate to the width
        // kernel only after a streak proves a long clear run (CJK under a dirty lead, e.g. Hangul)
        let mut streak = 0u32;
        loop {
            if i + 3 > n || !(0xE0..=0xEF).contains(&bytes[i]) {
                break;
            }
            let cp = (((bytes[i] & 0x0F) as u32) << 12)
                | (((bytes[i + 1] & 0x3F) as u32) << 6)
                | ((bytes[i + 2] & 0x3F) as u32);
            if bmp_set(cp as u16) {
                return (i, cp);
            }
            if STORE {
                // SAFETY: verbatim whole char.
                unsafe { raw_extend(out, bytes.as_ptr().add(i), 3, 3) };
            }
            i += 3;
            streak += 1;
            if streak == 8 {
                let (pos, cp2) = skip3::<STORE, SIMD>(bytes, i, out);
                if cp2 != 0 {
                    return (pos, cp2);
                }
                i = pos;
                break;
            }
        }
    }
}

// ── decompose: blob lookup + emit ─────────────────────────────────────────────────────────────────

/// Blob entry of `cp` under the form: `(off, len)` into the blob, `(0, 0)` = none (Hangul/inert).
#[inline]
fn blob_entry<const K: bool>(cp: u32) -> (usize, usize) {
    let (idx, data) = if K {
        (&NFKD_IDX, &NFKD_DATA[..])
    } else {
        (&NFD_IDX, &NFD_DATA[..])
    };
    let packed = data[(idx[(cp >> 6) as usize] + (cp & 63)) as usize];
    ((packed >> 8) as usize, (packed & 0xFF) as usize)
}
#[inline]
fn blob<const K: bool>() -> &'static [u8] {
    if K { &NFKD_BLOB } else { &NFD_BLOB }
}

/// Append `copy` bytes from `src` (over-copy rides the capacity slack) advancing by exactly `adv`.
#[inline(always)]
unsafe fn raw_extend(out: &mut String, src: *const u8, copy: usize, adv: usize) {
    // SAFETY (caller): capacity ≥ len + max(copy, adv); src has `copy` readable bytes; valid UTF-8.
    unsafe {
        let v = out.as_mut_vec();
        let len = v.len();
        debug_assert!(len + copy.max(adv) <= v.capacity());
        std::ptr::copy_nonoverlapping(src, v.as_mut_ptr().add(len), copy);
        v.set_len(len + adv);
    }
}

/// Emit the maximal run of decompose-relevant chars from `i` (first char pre-decoded as `cp`).
/// Hangul is an arithmetic subloop; everything else is one 16-byte blob copy guarded by the
/// rank-chain header; an out-of-order mark takes the cold in-place reorder. Returns the end position.
fn emit_run<const K: bool>(
    bytes: &[u8],
    mut i: usize,
    mut cp: u32,
    out: &mut String,
    last_rank: &mut u8,
    run_out: &mut usize,
) -> usize {
    let n = bytes.len();
    loop {
        if HANGUL.contains(&cp) {
            // syllable run: direct 3-byte decode per step, single 16-byte store per syllable
            loop {
                let s = cp - 0xAC00;
                let t = s % 28;
                let mut buf = [0u8; 16];
                for (slot, j) in [0x1100 + s / 588, 0x1161 + (s % 588) / 28, 0x11A7 + t]
                    .into_iter()
                    .enumerate()
                {
                    buf[slot * 3] = 0xE0 | (j >> 12) as u8;
                    buf[slot * 3 + 1] = 0x80 | ((j >> 6) & 0x3F) as u8;
                    buf[slot * 3 + 2] = 0x80 | (j & 0x3F) as u8;
                }
                // SAFETY: 16-byte stack buffer; capacity reserved.
                unsafe { raw_extend(out, buf.as_ptr(), 16, if t != 0 { 9 } else { 6 }) };
                i += 3;
                if i + 3 <= n && (0xEA..=0xED).contains(&bytes[i]) {
                    let c2 = (((bytes[i] & 0x0F) as u32) << 12)
                        | (((bytes[i + 1] & 0x3F) as u32) << 6)
                        | ((bytes[i + 2] & 0x3F) as u32);
                    if HANGUL.contains(&c2) {
                        cp = c2;
                        continue;
                    }
                }
                break;
            }
            *last_rank = 0;
            *run_out = out.len();
        } else {
            let (off, blen) = blob_entry::<K>(cp);
            if off == 0 {
                // identity mark (or a form-stable char the union bitmap flagged): verbatim
                let w = decode_cp(bytes, i).1;
                // SAFETY: exact in-bounds copy; capacity reserved.
                unsafe { raw_extend(out, bytes.as_ptr().add(i), w, w) };
                let r = rank(tag(cp));
                if r == 0 {
                    *last_rank = 0;
                    *run_out = out.len();
                } else if r >= *last_rank {
                    if *last_rank == 0 {
                        *run_out = out.len() - w;
                    }
                    *last_rank = r;
                } else {
                    // out-of-order mark: rollback this char, insert it into the run in rank order
                    let v = unsafe { out.as_mut_vec() };
                    unsafe { v.set_len(v.len() - w) };
                    reorder_insert(out, *run_out, r, &bytes[i..i + w]);
                }
                i += w;
            } else {
                let b = blob::<K>();
                let (first, last, mark_off) = (b[off - 3], b[off - 2], b[off - 1] as usize);
                if first != 0 && first < *last_rank {
                    // decomposition starts with an out-of-order mark (adversarial): insert char-wise
                    let mut pos = off;
                    let end = off + blen;
                    while pos < end {
                        let w = decode_cp(b, pos).1;
                        let (dcp, _) = decode_cp(b, pos);
                        let r = rank(tag(dcp));
                        if r == 0 {
                            // SAFETY: blob bytes are valid UTF-8; capacity reserved.
                            unsafe { raw_extend(out, b.as_ptr().add(pos), w, w) };
                            *last_rank = 0;
                            *run_out = out.len();
                        } else if r >= *last_rank {
                            if *last_rank == 0 {
                                *run_out = out.len();
                            }
                            unsafe { raw_extend(out, b.as_ptr().add(pos), w, w) };
                            *last_rank = r;
                        } else {
                            reorder_insert(out, *run_out, r, &b[pos..pos + w]);
                        }
                        pos += w;
                    }
                } else {
                    let pos0 = out.len();
                    // SAFETY: 16-byte-rounded copy from the padded blob (overshoot ≤ 15 rides the
                    // capacity slack); long compat expansions (e.g. `㈝` → 17 bytes) stay one copy.
                    unsafe { raw_extend(out, b.as_ptr().add(off), (blen + 15) & !15, blen) };
                    *last_rank = last;
                    if mark_off > 0 {
                        *run_out = pos0 + mark_off;
                    }
                }
                i += decode_cp(bytes, i).1;
            }
        }
        // continue while the next char is decompose-relevant (dense runs stay here)
        if i >= n || bytes[i] < 0x80 {
            return i;
        }
        let (ncp, _) = decode_cp(bytes, i);
        let nt = tag(ncp);
        if !(d_breaks::<K>(nt) || rank(nt) != 0 || nt & 0x3F == 0x3D) {
            return i;
        }
        cp = ncp;
    }
}

/// Insert mark bytes `ch` (rank `r`) into the canonically-ordered mark run at `out[run_out..]`,
/// before the first mark whose rank exceeds `r`. Cold: only out-of-canonical-order input hits this.
#[cold]
fn reorder_insert(out: &mut String, run_out: usize, r: u8, ch: &[u8]) {
    let mut pos = out.len();
    let mut byte = run_out;
    for c in out[run_out..].chars() {
        if rank(tag(c as u32)) > r {
            pos = byte;
            break;
        }
        byte += c.len_utf8();
    }
    // insert_str on a String is O(n-pos) but the run is a handful of marks
    out.insert_str(pos, unsafe { std::str::from_utf8_unchecked(ch) });
}

/// Decompose under the form. Single forward pass: the check IS the scan — on the first char that
/// breaks the form, rewind to the enclosing starter, copy the verified prefix wholesale, and continue
/// with the same kernels in write-through mode.
pub(crate) fn decompose<const K: bool, const SIMD: bool>(input: &str) -> Cow<'_, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    const MAX_EXPAND: usize = 4; // ≤ 3 in practice (Hangul); headroom keeps the reserve trivial
    let fb = d_bit::<K>();
    let mut dummy = String::new(); // check mode never writes
    let mut i = 0;
    let mut prev_rank = 0u8;
    let brk = loop {
        let (pos, cp) = match fb {
            1 => next_suspect::<1, false, SIMD>(bytes, i, &mut dummy),
            _ => next_suspect::<2, false, SIMD>(bytes, i, &mut dummy),
        };
        if pos >= n {
            return Cow::Borrowed(input);
        }
        if pos > i {
            prev_rank = 0; // stable chars intervened
        }
        let t = tag(cp);
        if d_breaks::<K>(t) || t & 0x3F == 0x3D {
            break pos; // changes under the form (0x3D: rank unknown — conservative)
        }
        let r = rank(t);
        if r != 0 {
            if r < prev_rank {
                break pos; // mark out of canonical order
            }
            prev_rank = r;
        } else {
            prev_rank = 0;
        }
        i = pos
            + if cp < 0x80 {
                1
            } else if cp < 0x800 {
                2
            } else if cp < 0x10000 {
                3
            } else {
                4
            };
    };
    // rewind over the in-order marks adjacent to the break so the rank chain restarts exactly
    let mut s = brk;
    while s > 0 {
        let mut p = s - 1;
        while p > 0 && bytes[p] & 0xC0 == 0x80 {
            p -= 1;
        }
        if rank(tag(decode_cp(bytes, p).0)) == 0 {
            break;
        }
        s = p;
    }
    let mut out = String::with_capacity(n * MAX_EXPAND + 48);
    out.push_str(&input[..s]);
    let (mut last_rank, mut run_out) = (0u8, out.len());
    let mut i = s;
    while i < n {
        let (pos, cp) = match fb {
            1 => next_suspect::<1, true, SIMD>(bytes, i, &mut out),
            _ => next_suspect::<2, true, SIMD>(bytes, i, &mut out),
        };
        if pos > i {
            last_rank = 0; // stable chars were written through
            run_out = out.len();
        }
        if pos >= n {
            break;
        }
        let t = tag(cp);
        // union-set but irrelevant under this form (e.g. compat starter under NFD): verbatim
        if !(d_breaks::<K>(t) || rank(t) != 0 || t & 0x3F == 0x3D) {
            let w = decode_cp(bytes, pos).1;
            // SAFETY: exact in-bounds copy; capacity reserved.
            unsafe { raw_extend(&mut out, bytes.as_ptr().add(pos), w, w) };
            last_rank = 0;
            run_out = out.len();
            i = pos + w;
            continue;
        }
        i = emit_run::<K>(bytes, pos, cp, &mut out, &mut last_rank, &mut run_out);
    }
    Cow::Owned(out)
}

// ── compose: pair lookup inside on-demand windows ─────────────────────────────────────────────────

#[inline]
fn composite(a: u32, b: u32) -> Option<u32> {
    if (0x1100..=0x1112).contains(&a) && (0x1161..=0x1175).contains(&b) {
        return Some(0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28);
    }
    if HANGUL.contains(&a) && (a - 0xAC00) % 28 == 0 && (0x11A8..=0x11C2).contains(&b) {
        return Some(a + (b - 0x11A7));
    }
    let key = ((a as u64) << 21) | b as u64;
    COMPOSE
        .binary_search_by_key(&key, |&(k, _)| k)
        .ok()
        .map(|p| COMPOSE[p].1)
}

/// Recompose `window` (UAX #15): fully decompose to `(rank, char)` pairs, canonically order, then
/// combine each starter with following non-blocked composables. Cold: only runs around actually
/// composition-relevant chars.
fn recompose_window<const K: bool>(window: &str, out: &mut String) {
    let mut pairs: Vec<(u8, char)> = Vec::with_capacity(window.len());
    let mut pending: Vec<(u8, char)> = Vec::new();
    let push = |r: u8, c: char, pairs: &mut Vec<(u8, char)>, pending: &mut Vec<(u8, char)>| {
        if r == 0 {
            if pending.len() > 1 {
                pending.sort_by_key(|&(rr, _)| rr);
            }
            pairs.append(pending);
            pairs.push((0, c));
        } else {
            pending.push((r, c));
        }
    };
    for c in window.chars() {
        let cp = c as u32;
        if HANGUL.contains(&cp) {
            let s = cp - 0xAC00;
            for (k, j) in [0x1100 + s / 588, 0x1161 + (s % 588) / 28, 0x11A7 + s % 28]
                .into_iter()
                .enumerate()
            {
                if k == 2 && s % 28 == 0 {
                    break;
                }
                push(0, char::from_u32(j).unwrap(), &mut pairs, &mut pending);
            }
            continue;
        }
        let (off, blen) = blob_entry::<K>(cp);
        if off == 0 {
            push(rank(tag(cp)), c, &mut pairs, &mut pending);
        } else {
            let b = blob::<K>();
            let mut p = off;
            while p < off + blen {
                let (dcp, w) = decode_cp(b, p);
                push(
                    rank(tag(dcp)),
                    char::from_u32(dcp).unwrap(),
                    &mut pairs,
                    &mut pending,
                );
                p += w;
            }
        }
    }
    if pending.len() > 1 {
        pending.sort_by_key(|&(rr, _)| rr);
    }
    pairs.append(&mut pending);
    // canonical composition
    let n = pairs.len();
    let mut i = 0;
    while i < n {
        let (r0, first) = pairs[i];
        if r0 != 0 {
            out.push(first); // stray leading mark
            i += 1;
            continue;
        }
        let mut cur = first as u32;
        let mut last_r: i16 = -1;
        let mut kept: Vec<char> = Vec::new();
        let mut j = i + 1;
        while j < n {
            let (r, c) = pairs[j];
            let not_blocked = if r == 0 {
                last_r == -1
            } else {
                (r as i16) > last_r || last_r == -1
            };
            if not_blocked && let Some(comp) = composite(cur, c as u32) {
                cur = comp;
                j += 1;
                continue;
            }
            if r == 0 {
                break;
            }
            last_r = r as i16;
            kept.push(c);
            j += 1;
        }
        out.push(char::from_u32(cur).unwrap());
        out.extend(kept);
        i = j;
    }
}

/// Compose under the form: scan with the same skip kernels; each composition-relevant hit opens a
/// window `[enclosing starter, end of the active cluster)` that is decomposed + recomposed; the rest
/// of the text — the overwhelming majority — is copied verbatim (or borrowed outright).
pub(crate) fn compose<const K: bool, const SIMD: bool>(input: &str) -> Cow<'_, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let fb = c_bit::<K>();
    let mut dummy = String::new();
    let mut out = String::new(); // allocated lazily: the borrow path never allocates
    let (mut gap, mut i) = (0usize, 0usize);
    let mut prev_rank = 0u8;
    while i < n {
        let (pos, cp) = match fb {
            4 => next_suspect::<4, false, SIMD>(bytes, i, &mut dummy),
            _ => next_suspect::<8, false, SIMD>(bytes, i, &mut dummy),
        };
        if pos >= n {
            break;
        }
        if pos > i {
            prev_rank = 0;
        }
        let t = tag(cp);
        let cw = decode_cp(bytes, pos).1;
        let relevant = if c_relevant::<K>(t) {
            true
        } else {
            let r = if t == 0x3C || t == 0x7E { 0 } else { rank(t) };
            if r != 0 {
                if r < prev_rank {
                    true // out of canonical order: recompose
                } else {
                    prev_rank = r;
                    false
                }
            } else {
                prev_rank = 0;
                false
            }
        };
        if !relevant {
            i = pos + cw;
            continue;
        }
        // NFKC fast write: neighbour-inert compat starter → baked composed blob, no window
        if K && t & 0x3F == 0x3C {
            let packed = NFKC_DATA[(NFKD_IDX[(cp >> 6) as usize] + (cp & 63)) as usize];
            let (off, blen) = ((packed >> 8) as usize, (packed & 0xFF) as usize);
            if (1..=16).contains(&blen) {
                if out.capacity() == 0 {
                    out.reserve(n + n / 2);
                }
                out.push_str(&input[gap..pos]);
                out.push_str(unsafe { std::str::from_utf8_unchecked(&NFKC_BLOB[off..off + blen]) });
                gap = pos + cw;
                i = gap;
                prev_rank = 0;
                continue;
            }
        }
        // window: rewind over the whole preceding cluster (any char that changes or is a mark) plus
        // one starter, then extend over the active cluster
        let mut ws = pos;
        while ws > gap {
            let mut p = ws - 1;
            while p > gap && bytes[p] & 0xC0 == 0x80 {
                p -= 1;
            }
            ws = p;
            let pt = tag(decode_cp(bytes, p).0);
            if !(d_breaks::<K>(pt) || rank(pt) != 0 || pt & 0x3F == 0x3D) {
                break; // reached the starter — include it and stop
            }
        }
        let mut seg = pos + cw;
        while seg < n {
            let (scp, sw) = decode_cp(bytes, seg);
            let st = tag(scp);
            if !(rank(st) != 0 || st & 0x40 != 0 || st == 0x7D || st & 0x3F == 0x3D) {
                break;
            }
            seg += sw;
        }
        if out.capacity() == 0 {
            out.reserve(n + n / 2);
        }
        out.push_str(&input[gap..ws]);
        recompose_window::<K>(&input[ws..seg], &mut out);
        gap = seg;
        i = seg;
        prev_rank = 0;
    }
    if gap == 0 {
        return Cow::Borrowed(input);
    }
    out.push_str(&input[gap..n]);
    Cow::Owned(out)
}

/// NFD of a single char via the baked tables (Hangul arithmetic; blob otherwise), in canonical order.
pub(crate) fn nfd_char(c: char, mut f: impl FnMut(char)) {
    let cp = c as u32;
    if HANGUL.contains(&cp) {
        let s = cp - 0xAC00;
        for (k, j) in [0x1100 + s / 588, 0x1161 + (s % 588) / 28, 0x11A7 + s % 28]
            .into_iter()
            .enumerate()
        {
            if k == 2 && s % 28 == 0 {
                break;
            }
            f(char::from_u32(j).unwrap());
        }
        return;
    }
    if cp < 0x30000 {
        let (off, blen) = blob_entry::<false>(cp);
        if blen != 0 {
            let b = blob::<false>();
            let mut p = off;
            while p < off + blen {
                let (dcp, w) = decode_cp(b, p);
                f(char::from_u32(dcp).unwrap());
                p += w;
            }
            return;
        }
    }
    f(c);
}
