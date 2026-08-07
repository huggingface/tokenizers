//! The o200k family as one bitstream program: o200k_base / GPT-4o, Llama-4, gpt-oss, MiniMax,
//! Mistral tekken, and kimi-k2. Same skeleton as cl100k — only the letter half differs.
//!
//! Two things are worth knowing before reading the algebra:
//!
//!  1. **The case split is a scalar escape, on purpose.** `[UC]*[LC]+ | [UC]+[LC]*` has no local
//!     form: `中Qz` is ONE token (a U after a C is not a boundary) but `ʰABC` is two (it is, when no
//!     L follows) — the difference is whether an L appears LATER in the run, so no `p1` decides it.
//!     The bit half instead computes a cheap gate: a letter token is escaped only if the block holds
//!     an interior upper or a trailing apostrophe, so all-lowercase and Capitalised text never pays.
//!  2. **The contraction is a SUFFIX here, not an alternative.** cl100k emits `'t` as its own token;
//!     o200k glues it onto the letter token before it. So this uses `emit_contr_suffix`, which
//!     *extends* the open token instead of opening one — the mirror image of `emit_contr`.
//!
//! `AUX` doubles as the variant selector because the two text-derived streams are mutually
//! exclusive: the o200k/tekken line wants `/` for rule 4's `[\r\n/]*` tail, kimi wants `\p{Han}`
//! for its leading `[\p{Han}]+` arm and has a plain `[\r\n]*` tail.

use crate::{
    AUX_HAN, AUX_SLASH, CODE_CONT, CONT, Span, build_block, contr_len, digit_groups, fill_to_last,
    lead_run, scanthru, to_lead, trail_run,
};

/// Atom tag → dense 4-bit code. Unlike cl100k, `\p{M}` IS a letter here (both alt classes list
/// `\p{M}`) — but only a true mark: `AlphaSymMark` (0x16, categorically `\p{S}`) and `Zwj` (0x26,
/// `\p{Cf}`) stay "other", which is what keeps `[\p{L}\p{M}]+` off them.
const LUT: [u8; 64] = {
    let mut t = [8u8; 64]; // other = [^\s\p{L}\p{N}] (incl. Connector/Punct/Sym/Control/ASM/ZWJ)
    t[0x10] = 0; // \p{Lu} ∪ \p{Lt}
    t[0x20] = 1; // \p{Ll}
    t[0x00] = 2;
    t[0x06] = 2; // caseless letter (\p{Lm}\p{Lo}) ∪ \p{M} — in BOTH alt classes
    t[0x01] = 3;
    t[0x02] = 3; // \p{N}
    t[0x03] = 4; // Newline
    t[0x04] = 5; // Space
    t[0x05] = 6; // WsOther
    t[0x09] = 9; // Apostrophe — "other" for every run rule, split out for the contraction flag
    t[0x0F] = CODE_CONT;
    t
};

struct Cls {
    u: u64,
    l: u64,
    c: u64,
    n: u64,
    nl: u64,
    sp: u64,
    ws: u64,
    other: u64,
    apo: u64,
}

#[inline]
fn decode(p0: u64, p1: u64, p2: u64, p3: u64, valid: u64) -> Cls {
    let low = !p3;
    let a = low & !p2;
    let w = low & p2;
    Cls {
        u: a & !p1 & !p0 & valid, // code 0 — past the block end every plane reads 0, hence `valid`
        l: a & !p1 & p0,
        c: a & p1 & !p0,
        n: a & p1 & p0,
        nl: w & !p1 & !p0,
        sp: w & !p1 & p0,
        ws: w & !(p1 & p0), // codes 4,5,6 — cont (7) excluded
        other: p3,          // codes 8,9 — the apostrophe is "other" for run purposes
        apo: p3 & p0,
    }
}

const C_U: u16 = 1 << 0;
const C_L: u16 = 1 << 1;
const C_C: u16 = 1 << 2;
const C_N: u16 = 1 << 3;
const C_NL: u16 = 1 << 4;
const C_SP: u16 = 1 << 5;
const C_WSO: u16 = 1 << 6;
const C_OTH: u16 = 1 << 7;
const C_HAN: u16 = 1 << 8;
const C_WS: u16 = C_NL | C_SP | C_WSO;
const C_LET: u16 = C_U | C_L | C_C;

const fn code_bits(code: u8) -> u16 {
    match code {
        0 => C_U,
        1 => C_L,
        2 => C_C,
        3 => C_N,
        4 => C_NL,
        5 => C_SP,
        6 => C_WSO,
        8 | 9 => C_OTH,
        _ => 0, // cont never reaches an edge (the fill resolves it)
    }
}

/// o200k_base / GPT-4o — and byte-for-byte the same regex Llama-4, gpt-oss and MiniMax-M2 ship.
#[must_use]
pub fn bitsplit_o200k(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    o200k::<{ AUX_SLASH }, true, 3>(text, tags, starts, flag, out)
}

/// Mistral tekken: o200k with no contraction suffix and one token per digit.
#[must_use]
pub fn bitsplit_tekken(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    o200k::<{ AUX_SLASH }, false, 1>(text, tags, starts, flag, out)
}

/// kimi-k2 / k3: o200k plus a leading `[\p{Han}]+` alternative, Han subtracted from both letter
/// classes, and a plain `[\r\n]*` rule-4 tail.
#[must_use]
pub fn bitsplit_kimi(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    o200k::<{ AUX_HAN }, true, 3>(text, tags, starts, flag, out)
}

fn o200k<const AUX: u8, const CONTRACTION: bool, const DIGIT_CAP: usize>(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    let ntext = text.len();
    if ntext == 0 {
        return 0;
    }
    let nblk = ntext.div_ceil(64);
    assert!(
        tags.len() >= ntext && starts.len() >= nblk && flag.len() >= nblk && out.len() >= ntext
    );
    let han_arm = AUX == AUX_HAN;

    let (mut code, mut prev_cont) = (CODE_CONT, 0u64);
    let (mut prev_han, mut nl_run, mut prev_osf) = (false, false, false);
    let (mut dig_run, mut dig_since) = (false, 0u32);
    let mut anl: Option<usize> = None;
    let mut last_lt: Option<usize> = None;

    for bi in 0..nblk {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let last_blk = base + len == ntext;

        let (b, last_code) =
            build_block::<AUX, true>(text, tags, base, len, &LUT, code, prev_han);
        let c = decode(b.p0, b.p1, b.p2, b.p3, valid);

        let han = if han_arm { b.aux & valid } else { 0 };
        let slash = if AUX == AUX_SLASH { b.aux } else { 0 };
        let last_han = han_arm && han >> (len - 1) & 1 != 0;

        let pb = if base == 0 {
            0
        } else {
            code_bits(code) | if prev_han { C_HAN } else { 0 }
        };
        let (nb, nb_lead) = if last_blk {
            (0u16, true)
        } else {
            let q = base + len;
            let is_lead = tags[q] != CONT;
            let bits = if is_lead {
                code_bits(LUT[tags[q] as usize])
                    | if han_arm && crate::aux_at::<{ AUX_HAN }>(text, q) {
                        C_HAN
                    } else {
                        0
                    }
            } else {
                code_bits(last_code) | if last_han { C_HAN } else { 0 }
            };
            (bits, is_lead)
        };
        let has = |v: u16, s: u16| v & s != 0;
        let p1 = |x: u64, k: bool| (x << 1) | u64::from(k);
        let n1 = |x: u64, k: bool| (x >> 1) | (u64::from(k) << 63);

        let lead = valid & !b.cont;
        let lb = ((lead >> 1) & valid) | (u64::from(nb_lead) << (len - 1));
        let eof_bit = if last_blk { 1u64 << (len - 1) } else { 0 };

        // ── Han is peeled off the letter classes first: kimi's alt-1 outranks the letter alts, and
        // its letter classes are literally `[…&&[^\p{Han}]]`. With AUX != HAN this is all zero.
        let (cu, cl, cc) = (c.u & !han, c.l & !han, c.c & !han);
        let letter = cu | cl | cc;
        let pb_let = has(pb, C_LET) && !has(pb, C_HAN);

        // ── rule 4 ` ?[^\s\p{L}\p{N}]+[\r\n/]*` and rules 5-7 — identical to cl100k.
        let o_start = c.other & lead & !p1(c.other, has(pb, C_OTH)) & !p1(c.sp, has(pb, C_SP));
        let ws_start = c.ws & lead & !p1(c.ws, has(pb, C_WS));
        let (steal, steal_patch) = to_lead(
            c.ws & !c.nl & lb & !eof_bit & !n1(c.ws, has(nb, C_WS)),
            b.cont,
            prev_cont,
        );
        let after_nl = p1(c.nl, has(pb, C_NL))
            & c.ws
            & lead
            & !fill_to_last(c.nl.reverse_bits(), c.ws.reverse_bits()).reverse_bits();

        // ── `[^\r\n\p{L}\p{N}]?` before a letter run: any token-opening non-newline non-letter
        // non-digit char. Smeared across its char's bytes so the test is a plain `p1` (see cl100k).
        let mut osf = o_start;
        osf |= (osf << 1) & b.cont;
        osf |= (osf << 2) & b.cont & (b.cont << 1);
        if prev_osf {
            osf |= lead_run(b.cont, valid);
        }
        let l_start = letter
            & lead
            & !p1(letter, pb_let)
            & !p1(c.ws & !c.nl, has(pb, C_WS) && !has(pb, C_NL))
            & !p1(osf, prev_osf);
        // the token holding a letter run may OPEN one char earlier, on the `[^\r\n\p{L}\p{N}]?`
        // prefix — that char is the start the escape has to be handed. The class genuinely excludes
        // `\r\n` and digits: a newline before a letter run is its own token, never a prefix.
        let prefix_cls = c.other | c.sp | (c.ws & !c.nl);
        let l_run = letter & lead & !p1(letter, pb_let);
        let next_l_run = nb_lead
            && has(nb, C_LET)
            && !has(nb, C_HAN)
            && letter >> (len - 1) & 1 == 0;
        // `n1` only reads the NEXT char at a char's last byte, so mark there and walk back to the
        // lead — a 3-byte prefix char (ZWJ, `—`, `½`) otherwise reads its own middle byte.
        let (pfx_lead, pfx_patch) =
            to_lead(prefix_cls & lb & n1(l_run, next_l_run), b.cont, prev_cont);
        let l_start_tok = l_start | pfx_lead;

        // ── the escape gate (see the header). An interior upper, or an apostrophe closing the
        // run, means some letter token in this block needs the scalar case/contraction pass.
        let interior_u = cu & lead & p1(letter, pb_let);
        let apo_after = c.apo & lead & p1(letter, pb_let);

        // ── kimi's `[\p{Han}]+`
        let han_start = han & lead & !p1(han, has(pb, C_HAN));

        // ── rule 3 `\p{N}{1,DIGIT_CAP}`
        let groups =
            digit_groups::<DIGIT_CAP>(c.n, lead, b.cont, has(pb, C_N), dig_run, dig_since);

        // ── the other-run's `[\r\n/]*` tail (`/` only for the o200k line; kimi has `[\r\n]*`).
        // The tail can only OPEN on a newline: `/` is in the `+` body, so a `/` after an other-run
        // was already eaten by the `+` (`\u{1f600}/!` is one token). It does continue the tail once
        // a newline has opened it, which is why `tail_cls` still carries it.
        let tail_cls = c.nl | slash;
        let nl_m = ((p1(c.other, has(pb, C_OTH)) & c.nl & lead) as u128) | u128::from(nl_run);
        let nl_e = scanthru(nl_m, tail_cls as u128);
        let nl_span = nl_e.wrapping_sub(nl_m);

        // ── the one backward-in-time rule, exactly as in cl100k/deepseek: a newline arriving now
        // retracts an "after the last newline" start committed for a run still open at the edge.
        if let Some(p) = anl
            && has(pb, C_WS)
            && c.ws & 1 != 0
            && c.nl & lead_run(c.ws, valid) != 0
        {
            starts[p / 64] &= !(1u64 << (p % 64));
            anl = None;
        }

        let mut st = groups | l_start | han_start | o_start | ws_start | after_nl | steal;
        st &= !(nl_span as u64);
        st |= nl_e as u64;
        st &= lead;
        if bi == 0 {
            st |= 1;
        }
        starts[bi] = st;
        // Conservative: flag every letter token in a block that shows either trigger. A false
        // positive only costs a rescan of that token, so erring wide is free; missing one is not.
        let lt = st & l_start_tok;
        let trig = (interior_u | apo_after) != 0;
        flag[bi] = if trig { lt } else { 0 };
        if trig && bi > 0 {
            flag[bi - 1] |= starts[bi - 1] & pfx_patch;
        }
        if bi > 0 {
            starts[bi - 1] |= steal_patch;
        }
        // a run open at the block edge may meet its trigger in a later block, so flag it now
        if lt != 0 {
            last_lt = Some(base + 63 - lt.leading_zeros() as usize);
        }
        // ...ending on the run's `?` prefix char counts too — the token opened there.
        let open_at_edge = (letter | l_start_tok) >> (len - 1) & 1 != 0;
        if !last_blk && open_at_edge && let Some(p) = last_lt {
            flag[p / 64] |= 1u64 << (p % 64);
        }

        // ── carries
        nl_run = nl_e >> 64 != 0;
        let tn = trail_run(c.n, valid, len);
        dig_run = tn != 0 && has(nb, C_N);
        dig_since = if !dig_run {
            0
        } else {
            let g = groups & tn;
            let counted = if g == 0 {
                dig_since + (c.n & lead & tn).count_ones()
            } else {
                (c.n & lead & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
            };
            if DIGIT_CAP == 0 || DIGIT_CAP >= 64 {
                counted
            } else {
                counted % DIGIT_CAP as u32
            }
        };
        let tws = trail_run(c.ws, valid, len);
        if tws != 0 && has(nb, C_WS) {
            let a = after_nl & tws & !(nl_e as u64);
            if a != 0 {
                anl = Some(base + 63 - a.leading_zeros() as usize);
            } else if !(tws & 1 != 0 && has(pb, C_WS)) {
                anl = None;
            }
        } else {
            anl = None;
        }
        prev_osf = osf >> (len - 1) & 1 != 0;
        prev_han = last_han;
        code = last_code;
        prev_cont = b.cont;

    }

    emit_o200k::<CONTRACTION>(text, tags, starts, flag, nblk, ntext, out)
}

// ── the scalar letter escape ────────────────────────────────────────────────────────────────────
// Ported from the FSM this replaces, which is the proven reading of the two letter alternatives.

const LET: u8 = 0x00; // coarse Atom::Letter (low nibble)
const MRK: u8 = 0x06; // coarse Atom::Mark
const ASM: u8 = 0x16; // AlphaSymMark — coarse Mark, categorically \p{S}
const ZWJ: u8 = 0x26; // ZWJ/ZWNJ — coarse Mark, categorically \p{Cf}

/// A real `[\p{L}\p{M}]` member. ALPHA_SYM and ZWJ are coarse `Mark` but neither `\p{L}` nor
/// `\p{M}`, so they are NOT letters here — that is what keeps them on the rule-4 path.
#[inline]
fn member(t: u8) -> bool {
    let c = t & 0x0F;
    c == LET || (c == MRK && t != ASM && t != ZWJ)
}

/// One letter sub-token from `p` within the run `[.., re)`: alt-1 `[UC]*[LC]+` (tried first), else
/// alt-2 `[UC]+[LC]*` for an all-upper run. Greedy with Perl backtracking — `[UC]*` gives back to
/// the last C so `[LC]+` can take one. Byte-wise: `Cont` is neither U nor L, so it rides along.
#[inline]
fn letter_match(tags: &[u8], p: usize, re: usize) -> usize {
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

/// `emit` plus the letter escape: at a flagged token, split the letter run into case sub-tokens and
/// let the last one absorb a trailing contraction. Everything else walks the plain start bits.
fn emit_o200k<const CONTRACTION: bool>(
    text: &[u8],
    tags: &[u8],
    starts: &[u64],
    flag: &[u64],
    nblk: usize,
    n: usize,
    out: &mut [Span],
) -> usize {
    let (mut w, mut open, mut skip) = (0usize, usize::MAX, 0usize);
    let mut open_flagged = false;
    let mut close = |w: &mut usize, open: usize, end: usize, flagged: bool| -> usize {
        // the run starts at `open`, or one char later when the token opened on the `?` prefix
        let ls = if !flagged || member(tags[open]) {
            open
        } else {
            open + crate::classify::char_len(text[open])
        };
        if !flagged || ls >= end {
            out[*w] = Span::new(open as u32, end as u32);
            *w += 1;
            return end;
        }
        let (mut p, mut first, mut cursor) = (ls, true, end);
        while p < end {
            let e = letter_match(tags, p, end);
            let start = if first { open } else { p };
            let te = if CONTRACTION && e == end {
                e + contr_len(text, e, true)
            } else {
                e
            };
            out[*w] = Span::new(start as u32, te as u32);
            *w += 1;
            first = false;
            cursor = te;
            p = e;
        }
        cursor
    };

    for bi in 0..nblk {
        let mut m = starts[bi];
        let f = flag[bi];
        while m != 0 {
            let j = m.trailing_zeros() as usize;
            let pos = bi * 64 + j;
            m &= m - 1;
            if pos < skip {
                continue;
            }
            if open != usize::MAX {
                let cur = close(&mut w, open, pos, open_flagged);
                if cur > pos {
                    // A contraction was absorbed: reopen past it and drop the start bit there. The
                    // reopened token can itself need the escape (`'ll'llA`), so read its flag —
                    // the loop will never see that start bit again.
                    open = if cur < n { cur } else { usize::MAX };
                    // `cur` lands mid letter-run, a position the algebra never marked as a start,
                    // so there is no flag bit to read there — a letter always needs the escape.
                    open_flagged = cur < n
                        && (member(tags[cur]) || flag[cur / 64] >> (cur % 64) & 1 != 0);
                    skip = cur + 1;
                    continue;
                }
            }
            open = pos;
            open_flagged = f >> j & 1 != 0;
        }
    }
    if open != usize::MAX {
        close(&mut w, open, n, open_flagged);
    }
    w
}
