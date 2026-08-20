//! `bitsplit` — GPT-family pre-tokenization as a **bitstream program**, replacing the scalar FSMs.
//!
//! Follows *Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs*
//! (MICRO'25, doi 10.1145/3725843.3756052). The paper's two ideas that carry over to a CPU:
//!
//!   1. **Bit-parallel regex.** Compile the grammar into character-class *bitstreams* (one bit per
//!      input byte) plus boolean ops and carry-propagating adds. 64 input bytes are decided per
//!      64-bit register op, branchlessly — the FSM's per-token unpredictable branch disappears.
//!   2. **Interleaved execution.** Do NOT run one loop per bitstream instruction over the whole
//!      input (that is the "sequential" baseline the paper beats: every intermediate stream is
//!      materialised and re-read). Instead fuse *all* instructions into ONE block-wise loop, so an
//!      intermediate stream lives in a register for the ~100 ops it is needed and dies there. The
//!      only thing that reaches memory is the final `starts` bitmap (n/8 bytes).
//!
//! The paper's third contribution (dependency-aware thread-data mapping) is a GPU concern — it
//! resolves cross-block dependencies by recomputing them on other SMs. On one core the blocks are
//! visited in order, so those dependencies are carried in a handful of scalar registers and, for
//! the one genuinely backward-in-time rule, patched into the already-written bitmap.
//!
//! We do not re-derive character classes: [`classify`] already emits one `Atom` tag per byte. The
//! builder folds those 16 atoms into a grammar-specific **dense 3-bit code** and extracts 3
//! bit-planes of it; every class stream is then a 2–3 op boolean function of the planes.
//!
//! Grammars: [`bitsplit_deepseek`], [`bitsplit_byte_level`] (GPT-2), [`bitsplit_cl100k`]. All three
//! byte-exact with the oniguruma oracle over a block-phase sweep — see `tests/parity.rs`.

/// Declares a grammar's block-local class streams, and with them the two carried shifts every
/// rule needs: `back` — what held one position earlier, carrying in the previous block's last
/// bit — and `fwd`, one position later, carrying in the next block's first bit.
///
/// This is the whole cross-block story. Keeping the carry in the *stream* instead of passing it at
/// every call site is what lets a rule read as plain stream algebra: "a run starts where this class
/// holds and the position before it did not" is `x & lead & !bk.x`, exact at a block edge. Before
/// this, each grammar rebuilt the edge byte's class into a `u16` mask (`code_bits`/`bits_at`) and
/// threaded it through every shift by hand — so the boundary byte got classified twice, once by the
/// builder and again scalar-side.
macro_rules! streams {
    ($(#[$m:meta])* $name:ident { $($f:ident),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Default)]
        struct $name { $($f: u64,)+ }

        impl $name {
            /// Every stream shifted one position later, carrying in `p`'s last bit.
            #[inline]
            fn back(&self, p: &Self) -> Self {
                Self { $($f: (self.$f << 1) | (p.$f >> 63),)+ }
            }

            /// Every stream shifted one position earlier, carrying in `n`'s first bit.
            #[inline]
            fn fwd(&self, n: &Self) -> Self {
                Self { $($f: (self.$f >> 1) | (n.$f << 63),)+ }
            }
        }
    };
}

pub mod classes;
pub mod classify;
mod han;
pub mod literal;
pub mod models;
pub mod regexes;
mod simd;

pub use models::deepseek::bitsplit_deepseek;
pub use models::cl100k::{bitsplit_cl100k, bitsplit_qwen};
pub use models::gpt2::bitsplit_byte_level;
pub use models::kimi::bitsplit_kimi;
pub use models::o200k::bitsplit_o200k;
pub use models::tekken::bitsplit_tekken;

/// A token span: byte offsets `[start, end)` into the input. `#[repr(C)]` so the output buffer has a
/// stable `[start, end]` layout — the pipeline reuses it with zero conversion, and it can be
/// reinterpreted as bytes / handed across the crate boundary.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    #[inline]
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// `[start, end)` as a `usize` range — for slicing the input text.
    #[inline]
    #[must_use]
    pub fn range(self) -> core::ops::Range<usize> {
        self.start as usize..self.end as usize
    }
}

impl PartialEq<(u32, u32)> for Span {
    fn eq(&self, other: &(u32, u32)) -> bool {
        self.start == other.0 && self.end == other.1
    }
}

/// Whether this target builds its bitstreams with SIMD. Without it the builder is the portable
/// byte-at-a-time reference, which is slower than the FSM this replaces -- so a caller should keep
/// its FSM path rather than route here. On x86 the kernel needs SSSE3, so this is a runtime check.
#[must_use]
pub fn fast_builder() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_ssse3()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn has_ssse3() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0);
    match CACHED.load(Ordering::Relaxed) {
        0 => {
            let yes = std::arch::is_x86_feature_detected!("ssse3");
            CACHED.store(1 + u8::from(yes), Ordering::Relaxed);
            yes
        }
        n => n == 2,
    }
}

pub(crate) const CONT: u8 = 15; // Atom::Cont
pub(crate) const CODE_CONT: u8 = 7; // every grammar's dense code for a continuation byte

/// What a grammar wants in the text-derived `Blk::aux` stream. Tag-derived classes go through the
/// LUT; these three need the raw bytes, so they are the one thing the builder reads `text` for.
pub(crate) const AUX_NONE: u8 = 0;
pub(crate) const AUX_CJK: u8 = 1; // deepseek Split-2: Han U+4E00..9FA5 ∪ kana U+3040..30FF
pub(crate) const AUX_SLASH: u8 = 2; // o200k rule 4's `[\r\n/]*` tail
pub(crate) const AUX_HAN: u8 = 3; // kimi-k2's leading `[\p{Han}]+` arm

/// The bitstreams for one 64-byte block. `p0`..`p3` are the bit-planes of the **filled** dense
/// code — filled meaning a multi-byte char sets its bits on *all* of its bytes, so "previous char's
/// class" is a plain `<< 1` and no rule does char-width arithmetic. `cont` is the one un-filled
/// stream (it defines the fill); `aux` is text-derived and only built when a grammar asks for it.
///
/// `p3` exists because o200k needs 9 tag classes (U/L/C/N/NL/SP/WSO/OTHER + cont) and code 7 is
/// reserved — the SIMD kernels find continuation lanes by testing `lut[tag] == 7`, so "other"
/// cannot be the leftover code. It is const-gated off for the 3-plane grammars.
#[derive(Default, Clone, Copy)]
pub(crate) struct Blk {
    pub cont: u64,
    pub p0: u64,
    pub p1: u64,
    pub p2: u64,
    pub p3: u64,
    pub aux: u64,
}

/// deepseek Split-2's isolated range: Han U+4E00..9FA5 ∪ Hiragana/Katakana U+3040..30FF (all
/// 3-byte, leads E3..E9). Same predicate as `fsm_deepseek`'s `ds_is_cjk_at`.
#[inline]
pub(crate) fn is_cjk_at(text: &[u8], p: usize) -> bool {
    let b = text[p];
    if !(0xE3..=0xE9).contains(&b) || p + 2 >= text.len() {
        return false;
    }
    let cp = ((b as u32 & 0x0F) << 12)
        | ((text[p + 1] as u32 & 0x3F) << 6)
        | (text[p + 2] as u32 & 0x3F);
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}

/// The `AUX` predicate at a lead byte.
#[inline]
pub(crate) fn aux_at<const AUX: u8>(text: &[u8], p: usize) -> bool {
    match AUX {
        AUX_CJK => is_cjk_at(text, p),
        AUX_SLASH => text[p] == b'/',
        AUX_HAN => crate::han::is_han_at(text, p),
        _ => false,
    }
}

/// Build one block. Full blocks go through the NEON kernel; the ragged tail (and every other
/// target) uses the portable byte-at-a-time reference. Both produce the identical `Blk`.
/// `lut` is the grammar's tag → dense code table.
#[inline]
pub(crate) fn build_block<const AUX: u8, const P3: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    len: usize,
    lut: &[u8; 64],
    cur_code: u8,
    cur_aux: bool,
) -> (Blk, u8) {
    #[cfg(target_arch = "aarch64")]
    if len == 64 {
        // SAFETY: `len == 64` means `base + 64 <= text.len() == tags.len()`.
        return unsafe {
            crate::simd::neon::build64::<AUX, P3>(text, tags, base, lut, cur_code, cur_aux)
        };
    }
    #[cfg(target_arch = "x86_64")]
    if len == 64 && has_ssse3() {
        // SAFETY: `len == 64` bounds both reads, and SSSE3 is checked above.
        return unsafe {
            crate::simd::x86::build64::<AUX, P3>(text, tags, base, lut, cur_code, cur_aux)
        };
    }
    build_block_scalar::<AUX, P3>(text, tags, base, len, lut, cur_code, cur_aux)
}

/// Portable reference builder: one byte at a time.
pub(crate) fn build_block_scalar<const AUX: u8, const P3: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    len: usize,
    lut: &[u8; 64],
    mut cur_code: u8,
    mut cur_aux: bool,
) -> (Blk, u8) {
    let mut b = Blk::default();
    for i in 0..len {
        let p = base + i;
        let bit = 1u64 << i;
        if tags[p] == CONT {
            b.cont |= bit;
        } else {
            cur_code = lut[tags[p] as usize];
            cur_aux = AUX != AUX_NONE && aux_at::<AUX>(text, p);
        }
        b.p0 |= bit * u64::from(cur_code & 1 != 0);
        b.p1 |= bit * u64::from(cur_code & 2 != 0);
        b.p2 |= bit * u64::from(cur_code & 4 != 0);
        if P3 {
            b.p3 |= bit * u64::from(cur_code & 8 != 0);
        }
        b.aux |= bit * u64::from(cur_aux);
    }
    (b, cur_code)
}

// ── bitstream primitives ────────────────────────────────────────────────────────────────────────

/// `ScanThru`: move every marker in `m` forward past the run of `c` it sits in, landing on the
/// first position not in `c`. The classic Parabix carry-propagation trick — one add. Done in `u128`
/// so a run reaching bit 63 puts its landing bit at 64 instead of vanishing: that bit *is* the
/// block's carry-out, and `e - m` still yields the right in-block span.
#[inline]
pub(crate) const fn scanthru(m: u128, c: u128) -> u128 {
    m.wrapping_add(c) & !c
}

/// Advance markers by ONE CHAR: shift one byte, then scan through the continuation bytes. Markers
/// that leave the block are dropped — their state is reconstructed from the scalar carry instead.
#[inline]
pub(crate) const fn adv(m: u64, cont: u64) -> u64 {
    scanthru((m as u128) << 1, cont as u128) as u64
}

/// Move each bit back to the lead byte of its char. Rules that look at the *next* char are stated
/// at a char's last byte; token starts must sit on leads. ≤3 steps (UTF-8 chars are ≤4 bytes).
/// Returns `(in-block, patch for the previous word)` — a char straddling the block boundary lands
/// its lead in the word we already wrote.
#[inline]
pub(crate) fn to_lead(x: u64, cont: u64, prev_cont: u64) -> (u64, u64) {
    let c = ((cont as u128) << 64) | prev_cont as u128;
    let mut y = (x as u128) << 64;
    for _ in 0..3 {
        let m = y & c;
        if m == 0 {
            break;
        }
        y = (y & !c) | (m >> 1);
    }
    ((y >> 64) as u64, y as u64)
}

/// In each run of `c`, fill from the run start through the LAST marker of `m` (`m ⊆ c`). Used on
/// reversed streams, where it answers "is there a newline at-or-after me, still inside this
/// whitespace run?". `(end - m) | m` rather than `end - m`: the latter only spans from the *first*
/// marker when a run holds several.
#[inline]
pub(crate) fn fill_to_last(m: u64, c: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let (m128, c128) = (m as u128, c as u128);
    (scanthru(m128, c128).wrapping_sub(m128) | m128) as u64
}

/// Run of `x` that starts at bit 0.
#[inline]
pub(crate) fn lead_run(x: u64, valid: u64) -> u64 {
    let z = !x & valid;
    if z == 0 {
        valid
    } else {
        (z & z.wrapping_neg()) - 1
    }
}

/// Run of `x` that ends at bit `len - 1` (0 if that bit is clear).
#[inline]
pub(crate) fn trail_run(x: u64, valid: u64, len: usize) -> u64 {
    if x & (1u64 << (len - 1)) == 0 {
        return 0;
    }
    let z = !x & valid;
    if z == 0 {
        valid
    } else {
        valid & !((1u64 << (64 - z.leading_zeros())) - 1)
    }
}

/// `\p{N}{1,cap}` — a group boundary every `cap` chars from the run start. The one non-local rule
/// in every tiktoken-family grammar, so it lives here rather than three times over.
///
/// `cap == 0` or `>= 64` means an unbounded `\p{N}+`: the run is one token and there is nothing to
/// do (and the shifts below would be UB). `dig_since` resumes a run that crossed the block edge —
/// re-masking with `n` at every hop matters, because the carry only says the byte *at* the edge was
/// a digit, not that the run survived it.
#[inline]
pub(crate) fn digit_groups(
    cap: usize,
    n: u64,
    lead: u64,
    cont: u64,
    prev_is_digit: bool,
    dig_run: bool,
    dig_since: u32,
) -> u64 {
    let mut m = n & lead & !((n << 1) | u64::from(prev_is_digit));
    if cap == 0 || cap >= 64 {
        return m; // an unbounded `\p{N}+`: the run is one token
    }
    let capu = cap as u32;
    if dig_run && prev_is_digit {
        let mut s = lead & lead.wrapping_neg() & n; // first lead of the block
        for _ in 0..((capu - dig_since % capu) % capu) {
            s = adv(s, cont) & n & lead;
        }
        m |= s;
    }
    let mut groups = m;
    if n & cont == 0 {
        // Fast path: every digit here is single-byte, so "CAP chars on" is a plain shift against a
        // mask asking that the skipped positions were digits too — what the `adv` chain checks.
        let mut nk = n;
        let mut k = 1;
        while k < cap {
            nk &= n << k;
            k += 1;
        }
        while m != 0 {
            m = (m << cap) & nk;
            groups |= m;
        }
    } else {
        while m != 0 {
            let mut e = m;
            for _ in 0..cap {
                e = adv(e, cont) & n & lead;
            }
            if e == 0 {
                break;
            }
            groups |= e;
            m = e;
        }
    }
    groups
}

// ── scalar helpers, shared by the grammars that need an escape ──────────────────────────────
use crate::classify::{Atom, in_mask, mask};


pub(crate) const LET: u8 = 0x00; // coarse Atom::Letter (low nibble)
pub(crate) const MRK: u8 = 0x06; // coarse Atom::Mark
pub(crate) const ASM: u8 = 0x16; // AlphaSymMark — coarse Mark, categorically \p{S}
pub(crate) const ZWJ: u8 = 0x26; // ZWJ/ZWNJ — coarse Mark, categorically \p{Cf}
pub(crate) const NW: u8 = 0x01;
pub(crate) const NO: u8 = 0x02;
pub(crate) const NLN: u8 = 0x03;
pub(crate) const SPC: u8 = 0x04;
pub(crate) const WSO: u8 = 0x05;

/// A real `[\p{L}\p{M}]` member. ALPHA_SYM and ZWJ are coarse `Mark` but neither `\p{L}` nor
/// `\p{M}`, so they are NOT letters — that is what keeps them on the rule-4 path.
#[inline]
pub(crate) fn member(t: u8) -> bool {
    let c = t & 0x0F;
    c == LET || (c == MRK && t != ASM && t != ZWJ)
}

#[inline]
pub(crate) fn run_end(tags: &[u8], mut i: usize, end: usize, m: u16) -> usize {
    let m = m | Atom::Cont.bit();
    while i < end && in_mask(tags[i], m) {
        i += 1;
    }
    i
}

/// `\s*[\r\n]+ | \s+(?!\S) | \s+`: through the last `\r\n` if any, else the whole run at EOF, else
/// give the final ws char back to whatever follows.
#[inline]
pub(crate) fn ws_tail(text: &[u8], tags: &[u8], i: usize, end: usize) -> usize {
    let re = run_end(tags, i, end, mask::WS);
    if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
        i + r + 1
    } else if re == end {
        re
    } else {
        let mut last = re - 1;
        while last > i && text[last] & 0xC0 == 0x80 {
            last -= 1;
        }
        if last > i { last } else { re }
    }
}

/// One letter sub-token from `p` within `[.., re)`: alt-1 `[UC]*[LC]+` (tried first), else alt-2
/// `[UC]+[LC]*` for an all-upper run. `[UC]*` gives back to the last C so `[LC]+` can take one.
#[inline]
pub(crate) fn letter_match(tags: &[u8], p: usize, re: usize) -> usize {
    let (mut q, mut last_c) = (p, usize::MAX);
    while q < re && tags[q] != 0x20 {
        if tags[q] != CONT && tags[q] != 0x10 {
            last_c = q;
        }
        q += 1;
    }
    if q < re {
        let mut e = q;
        while e < re && tags[e] != 0x10 {
            e += 1;
        }
        return e;
    }
    if last_c == usize::MAX {
        return re; // all upper → alt-2 takes the whole run
    }
    let mut e = last_c;
    while e < re && tags[e] != 0x10 {
        e += 1;
    }
    e
}


// ── emit ────────────────────────────────────────────────────────────────────────────────────────

/// `starts` bitmap → spans. Each set bit closes the previous token and opens the next; `tzcnt`
/// walks them at ~3 ops per token.
pub(crate) fn emit(starts: &[u64], nblk: usize, n: usize, out: &mut [Span]) -> usize {
    let (mut w, mut open) = (0usize, u32::MAX);
    for (bi, &word) in starts.iter().enumerate().take(nblk) {
        let mut m = word;
        while m != 0 {
            let pos = (bi * 64 + m.trailing_zeros() as usize) as u32;
            if open != u32::MAX {
                out[w] = Span::new(open, pos);
                w += 1;
            }
            open = pos;
            m &= m - 1;
        }
    }
    if open != u32::MAX {
        out[w] = Span::new(open, n as u32);
        w += 1;
    }
    w
}

/// Emit with a **scalar escape at flagged bits**: contractions (`'s 't 're 've 'm 'll 'd`) are
/// variable-length, case-optional and outrank every other alternative, which makes them miserable
/// in bit algebra and trivial here. `flag` marks the apostrophes that open a token; a block with
/// none takes the plain loop, so the escape costs one test per block on ordinary text.
///
/// A matched contraction overrides the algebra outright: it emits its own span and skips every
/// start bit inside it, which is how the letter alternative loses the tie (`'sx` → `'s`, `x`).
pub(crate) fn emit_contr(
    text: &[u8],
    starts: &[u64],
    flag: &[u64],
    nblk: usize,
    n: usize,
    ci: bool,
    out: &mut [Span],
) -> usize {
    let (mut w, mut open, mut skip) = (0usize, u32::MAX, 0usize);
    for bi in 0..nblk {
        let mut m = starts[bi];
        let f = flag[bi];
        if f == 0 && skip <= bi * 64 {
            while m != 0 {
                let pos = (bi * 64 + m.trailing_zeros() as usize) as u32;
                if open != u32::MAX {
                    out[w] = Span::new(open, pos);
                    w += 1;
                }
                open = pos;
                m &= m - 1;
            }
            continue;
        }
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            let pos = bi * 64 + j;
            m &= m - 1;
            if pos < skip {
                continue;
            }
            if open != u32::MAX {
                out[w] = Span::new(open, pos as u32);
                w += 1;
            }
            open = pos as u32;
            if f >> j & 1 != 0 {
                // Contractions chain (`'re've`, `y'all'd've`): the char after one is a token start
                // in its own right, so keep matching until one fails rather than handing control
                // back to the bit algebra — whose start bit there we are about to skip.
                let mut p = pos;
                while p < n {
                    let l = contr_len(text, p, ci);
                    if l == 0 {
                        break;
                    }
                    out[w] = Span::new(p as u32, (p + l) as u32);
                    w += 1;
                    p += l;
                }
                if p > pos {
                    open = if p < n { p as u32 } else { u32::MAX };
                    // `+ 1`: the algebra usually also has a start bit exactly at `p` (the letter
                    // run resumes there) and we have just opened it — consuming it again would
                    // emit an empty span. `contr_len(p) == 0` here, so nothing is lost.
                    skip = p + 1;
                }
            }
        }
    }
    if open != u32::MAX {
        out[w] = Span::new(open, n as u32);
        w += 1;
    }
    w
}


/// Byte length of the contraction at `i` (2 or 3), or 0. `ci` picks cl100k/o200k's `(?i:)` form
/// over GPT-2's case-sensitive one.
#[inline]
pub(crate) fn contr_len(text: &[u8], i: usize, ci: bool) -> usize {
    let n = text.len();
    if i + 1 >= n || text[i] != b'\'' || text[i + 1] >= 0x80 {
        return 0;
    }
    let c1 = if ci { text[i + 1] | 0x20 } else { text[i + 1] };
    match c1 {
        b's' | b't' | b'm' | b'd' => 2,
        b'r' | b'v' | b'l' if i + 2 < n && text[i + 2] < 0x80 => {
            let c2 = if ci { text[i + 2] | 0x20 } else { text[i + 2] };
            usize::from((matches!(c1, b'r' | b'v') && c2 == b'e') || (c1 == b'l' && c2 == b'l')) * 3
        }
        _ => 0,
    }
}

/// Builder-only cost probe: runs just the byte→bitstream transpose over every block and folds the
/// streams so nothing is optimised away. The difference against a full grammar is what the
/// bitstream program itself costs.
#[doc(hidden)]
#[must_use]
pub fn build_only(text: &[u8], tags: &[u8]) -> u64 {
    let (mut acc, mut code, mut cjk) = (0u64, CODE_CONT, false);
    for base in (0..text.len()).step_by(64) {
        let len = (text.len() - base).min(64);
        let (b, c) = build_block::<{ AUX_CJK }, false>(text, tags, base, len, &models::deepseek::LUT, code, cjk);
        code = c;
        cjk = b.aux >> (len - 1) & 1 != 0;
        acc ^= b.cont ^ b.p0 ^ b.p1 ^ b.p2 ^ b.aux;
    }
    acc
}

/// Convenience wrapper: classify + deepseek bitsplit over caller-owned scratch.
#[must_use]
pub fn pre_tokenize(text: &[u8], tags: &mut [u8], starts: &mut [u64], out: &mut [Span]) -> usize {
    classify::classify(text, tags);
    bitsplit_deepseek(text, tags, starts, out)
}
