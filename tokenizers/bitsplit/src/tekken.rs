//! Mistral **tekken** (mistral-small-4 / mistral-4):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|…)?`
//! `|` same with `+`/`*` swapped `| \p{N} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+`
//!
//! o200k's grammar with two changes: letter tokens take NO contraction suffix, and the digit rule
//! is a bare `\p{N}` — one token per digit.
//!

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
use crate::classify::{char_len, in_mask, mask};
use crate::{
    AUX_SLASH, CODE_CONT, CONT, Span, build_block, digit_groups, fill_to_last,
    lead_run, letter_match, member, run_end, scanthru, to_lead, trail_run, ws_tail,
};

/// Atom tag → dense 4-bit code. Unlike cl100k, `\p{M}` IS a letter here (both alt classes list
/// `\p{M}`) — but only a true mark: `AlphaSymMark` (0x16, categorically `\p{S}`) and `Zwj` (0x26,
/// `\p{Cf}`) stay "other", which is what keeps `[\p{L}\p{M}]+` off them.
const LUT: [u8; 64] = {
    let mut t = [8u8; 64]; // other = [^\s\p{L}\p{N}] (incl. Connector/Punct/Sym/Control/ASM/ZWJ)
    t[0x10] = 0; // \p{Lu} ∪ \p{Lt}
    t[0x20] = 1; // \p{Ll}
    t[0x00] = 2; // caseless letter (\p{Lm}\p{Lo}) — in BOTH alt classes
    t[0x06] = 10; // true \p{M}: a letter for the alts, but ALSO in rule 4's class (see `mark_adj`)
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
    oth: u64,
    mark: u64,
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
        oth: p3 & !p1,      // codes 8,9 — the apostrophe is "other" for run purposes
        mark: p3 & p1,      // code 10
        apo: p3 & p0 & !p1, // code 9
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
const C_MARK: u16 = 1 << 8;
const C_WS: u16 = C_NL | C_SP | C_WSO;
const C_LET: u16 = C_U | C_L | C_C | C_MARK;

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
        10 => C_MARK,
        _ => 0, // cont never reaches an edge (the fill resolves it)
    }
}

/// Mistral tekken.
#[must_use]
pub fn bitsplit_tekken(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    tekken(text, tags, starts, flag, out)
}

fn tekken(
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
    
    let (mut code, mut prev_cont) = (CODE_CONT, 0u64);
    let (mut nl_run, mut prev_osf) = (false, false);
    let mut prev_absorbed = false; // the block's last byte was eaten by a `[\r\n/]*` tail
    let (mut dig_run, mut dig_since) = (false, 0u32);
    let mut anl: Option<usize> = None;
    let mut last_lt: Option<usize> = None;
    let mut prev_start: Option<usize> = None;

    for bi in 0..nblk {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let last_blk = base + len == ntext;

        let (b, last_code) =
            build_block::<{ AUX_SLASH }, true>(text, tags, base, len, &LUT, code, false);
        let c = decode(b.p0, b.p1, b.p2, b.p3, valid);

        let slash = b.aux;
        
        let pb = if base == 0 {
            0
        } else {
            code_bits(code)
        };
        let (nb, nb_lead) = if last_blk {
            (0u16, true)
        } else {
            let q = base + len;
            let is_lead = tags[q] != CONT;
            let bits = if is_lead {
                code_bits(LUT[tags[q] as usize])
            } else {
                code_bits(last_code)
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
        let (cu, cl, cc) = (c.u, c.l, c.c);
        let letter = cu | cl | cc | c.mark;
        let pb_let = has(pb, C_LET);

        // Rule 4's `[\r\n/]*` tail. `/` is in BOTH the `+` body and the tail, so one tail run can
        // collect SEVERAL markers (`!\n/\n`: the `\n` after `!` and the `\n` after `/`). That is
        // exactly what `fill_to_last` is for -- `nl_e - nl_m` would span only from the last one.
        let tail_cls = c.nl | slash;
        let nl_m64 = ((p1(c.oth, has(pb, C_OTH)) & c.nl & lead) | u64::from(nl_run)) & tail_cls;
        let nl_m = nl_m64 as u128;
        let nl_e = scanthru(nl_m, tail_cls as u128);
        let nl_span = fill_to_last(nl_m64, tail_cls) as u128;

        // ── rule 4 ` ?[^\s\p{L}\p{N}]+[\r\n/]*` and rules 5-7 — identical to cl100k.
        // A char absorbed by a `[\r\n/]*` tail is NOT part of the `+` body, so an "other" after it
        // opens a fresh run (`\u{1f600}\r\n/#` is the tail, then `#` starts again).
        let o_prev = c.oth & !(nl_span as u64);
        let o_start = c.oth
            & lead
            & !(nl_span as u64) // a char INSIDE the tail never opens a run
            & !p1(o_prev, has(pb, C_OTH) && !prev_absorbed)
            & !p1(c.sp, has(pb, C_SP));
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
        let prefix_cls = c.oth | c.sp | (c.ws & !c.nl);
        let l_run = letter & lead & !p1(letter, pb_let);
        let next_l_run = nb_lead
            && has(nb, C_LET)
            && letter >> (len - 1) & 1 == 0;
        // `n1` only reads the NEXT char at a char's last byte, so mark there and walk back to the
        // lead — a 3-byte prefix char (ZWJ, `—`, `½`) otherwise reads its own middle byte.
        let (pfx_lead, pfx_patch) =
            to_lead(prefix_cls & lb & n1(l_run, next_l_run), b.cont, prev_cont);
        let l_start_tok = l_start | pfx_lead;

        // ── the escape gate (see the header). An interior upper, or an apostrophe closing the
        // run, means some letter token in this block needs the scalar case/contraction pass.
        let interior_u = cu & lead & p1(letter, pb_let);
        // no contraction suffix here, so an apostrophe after a letter is nothing special
        // `\p{M}` is in BOTH the letter classes and rule 4's `[^\s\p{L}\p{N}]`, and which one wins
        // depends on whether the punctuation before it STARTED the run (`!\u{301}a` is one token,
        // `!!\u{301}a` is two). Adjacency is the gate; the scalar pass resolves it. Real text hits
        // this via emoji + variation selector (U+FE0F is \p{Mn}).
        let mark_adj = (c.mark & lead & p1(c.oth, has(pb, C_OTH)))
            | (c.oth & lead & p1(c.mark, has(pb, C_MARK)));

        // ── rule 3 `\p{N}{1,DIGIT_CAP}`
        let groups =
            digit_groups(1, c.n, lead, b.cont, has(pb, C_N), dig_run, dig_since);

        // ── the other-run's `[\r\n/]*` tail (`/` only for the o200k line; kimi has `[\r\n]*`).
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

        let mut st = groups | l_start | o_start | ws_start | after_nl | steal;
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
        let trig = interior_u != 0;
        flag[bi] = if trig { lt } else { 0 };
        // mark_adj is rare, so be blunt: escape every token in the block, plus the one still open
        // from an earlier block (rule 4's ` ?` means the token can have started on a space).
        if mark_adj != 0 {
            flag[bi] |= st;
            if let Some(p) = prev_start {
                flag[p / 64] |= 1u64 << (p % 64);
            }
        }
        if st != 0 {
            prev_start = Some(base + 63 - st.leading_zeros() as usize);
        }
        if trig {
            if bi > 0 {
                flag[bi - 1] |= starts[bi - 1] & pfx_patch;
            }
            // the trigger can land in a LATER block than the token it belongs to (`\u{d55c}` ends
            // block k, its `'s` opens block k+1), so always re-flag the last letter token seen.
            if let Some(p) = last_lt {
                flag[p / 64] |= 1u64 << (p % 64);
            }
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
        prev_absorbed = nl_span >> (len - 1) & 1 != 0;
        let tn = trail_run(c.n, valid, len);
        dig_run = tn != 0 && has(nb, C_N);
        dig_since = if !dig_run {
            0
        } else {
            let g = groups & tn;
            let _counted = if g == 0 {
                dig_since + (c.n & lead & tn).count_ones()
            } else {
                (c.n & lead & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
            };
            0 // digit cap 1: every digit is its own group, so nothing carries
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
        code = last_code;
        prev_cont = b.cont;

    }

    emit_tekken(text, tags, starts, flag, nblk, ntext, out)
}

// ── the scalar escape ───────────────────────────────────────────────────────────────────────────
// Ported from the FSM this replaces, which is the proven reading of the alternatives. It runs from
// a flagged token start and RESYNCS the moment it lands on an algebra start bit again, so a
// divergent region costs a short scalar walk and nothing downstream.

/// Emit ONE token starting at `i` (letters emit several) and return the cursor past it.
fn step(
    text: &[u8],
    tags: &[u8],
    i: usize,
    end: usize,
    out: &mut [Span],
    w: &mut usize,
) -> usize {
    let is_lm = |p: usize| p < end && member(tags[p]);
    let letter_end = |mut p: usize| {
        while p < end && (tags[p] == CONT || member(tags[p])) {
            p += 1;
        }
        p
    };
    let other = |sp0: usize| {
        let mut p = run_end(tags, sp0, end, mask::NOT_WS_L_N);
        if p > sp0 {
            while p < end && (tags[p] == crate::NLN || text[p] == b'/') {
                p += char_len(text[p]);
            }
        }
        p
    };
    let emit1 = |a: usize, b: usize, out: &mut [Span], w: &mut usize| {
        out[*w] = Span::new(a as u32, b as u32);
        *w += 1;
    };
    // the letter alternatives, with the case split and the optional contraction suffix
    let letters = |pfx: usize, ls: usize, out: &mut [Span], w: &mut usize| -> usize {
        let re = letter_end(ls);
        let (mut p, mut first, mut cursor) = (ls, true, re);
        while p < re {
            let e = letter_match(tags, p, re);
            let start = if first { pfx } else { p };
            let te = e; // tekken has no contraction suffix
            out[*w] = Span::new(start as u32, te as u32);
            *w += 1;
            first = false;
            cursor = te;
            p = e;
        }
        cursor
    };

    let b = text[i];
    match tags[i] & 0x0F {
        crate::NW | crate::NO => {
            let (mut p, mut cnt) = (i, 0usize);
            while p < end && cnt < 1 && in_mask(tags[p], mask::NUMBER) {
                p += char_len(text[p]);
                cnt += 1;
            }
            emit1(i, p, out, w);
            p
        }
        crate::LET | crate::MRK => {
            if member(tags[i]) {
                return letters(i, i, out, w);
            }
            let a = i + char_len(b);
            if is_lm(a) {
                return letters(i, a, out, w);
            }
            let p = other(i);
            emit1(i, p, out, w);
            p
        }
        crate::SPC => {
            let a = i + 1;
            if is_lm(a) {
                return letters(i, a, out, w);
            }
            let p = other(a);
            let e = if p > a { p } else { ws_tail(text, tags, i, end) };
            emit1(i, e, out, w);
            e
        }
        crate::WSO => {
            let a = i + char_len(b);
            if is_lm(a) {
                return letters(i, a, out, w);
            }
            let e = ws_tail(text, tags, i, end);
            emit1(i, e, out, w);
            e
        }
        crate::NLN => {
            let e = ws_tail(text, tags, i, end);
            emit1(i, e, out, w);
            e
        }
        _ => {
            let a = i + char_len(b);
            if is_lm(a) {
                return letters(i, a, out, w);
            }
            let p = other(i);
            emit1(i, p, out, w);
            p
        }
    }
}

/// `emit` plus the escape: at a flagged start, hand over to the scalar dispatch and take back
/// control at the first position that is a start bit again.
fn emit_tekken(
    text: &[u8],
    tags: &[u8],
    starts: &[u64],
    flag: &[u64],
    nblk: usize,
    n: usize,
    out: &mut [Span],
) -> usize {
    let (mut w, mut open, mut skip) = (0usize, usize::MAX, 0usize);
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
            if f >> j & 1 != 0 {
                if open != usize::MAX && open < pos {
                    out[w] = Span::new(open as u32, pos as u32);
                    w += 1;
                }
                let mut c = pos;
                loop {
                    c = step(text, tags, c, n, out, &mut w);
                    if c >= n || starts[c / 64] >> (c % 64) & 1 != 0 {
                        break;
                    }
                }
                open = usize::MAX;
                skip = c;
                continue;
            }
            if open != usize::MAX {
                out[w] = Span::new(open as u32, pos as u32);
                w += 1;
            }
            open = pos;
        }
    }
    if open != usize::MAX {
        out[w] = Span::new(open as u32, n as u32);
        w += 1;
    }
    w
}
