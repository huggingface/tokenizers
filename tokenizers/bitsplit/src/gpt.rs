//! The two tiktoken-family grammars as bitstream programs.
//!
//! * [`bitsplit_byte_level`] — GPT-2 / Llama / Qwen:
//!   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! * [`bitsplit_cl100k`] — cl100k_base / Llama-3:
//!   `(?i:'s|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Both share one dense code table, and both hand contractions to the scalar escape in
//! [`crate::emit_contr`] rather than trying to express a variable-length case-optional literal
//! alternation in bit algebra.

use crate::{
    CODE_CONT, CONT, Span, adv, build_block, emit_contr, fill_to_last, lead_run, scanthru, to_lead,
    trail_run,
};

/// Atom tag → dense 3-bit code, shared by both grammars. Unlike deepseek's table, `Mark` is NOT a
/// letter here (`\p{L}` excludes it, so it belongs to the "other" class), and `Apostrophe` gets its
/// own code so the contraction escape can be flagged with one AND — it is still "other" for every
/// run rule, which `decode` restores.
const LUT: [u8; 64] = {
    let mut t = [2u8; 64]; // other = [^\s\p{L}\p{N}]
    t[0x00] = 0;
    t[0x10] = 0;
    t[0x20] = 0; // Letter (+ case refinements)
    t[0x01] = 1;
    t[0x02] = 1; // \p{N} = Nd ∪ Nl ∪ No
    t[0x03] = 3; // Newline
    t[0x04] = 4; // Space
    t[0x05] = 5; // WsOther
    t[0x09] = 6; // Apostrophe
    t[0x0F] = CODE_CONT;
    t
};

/// One block's class streams. `other` folds the apostrophe code back in; only `l` needs the
/// `valid` mask, since past the block end every plane reads 0, i.e. code 0.
struct Cls {
    l: u64,
    n: u64,
    other: u64,
    nl: u64,
    sp: u64,
    ws: u64,
    apo: u64,
}

#[inline]
fn decode(p0: u64, p1: u64, p2: u64, valid: u64) -> Cls {
    let a = !p2 & !p1;
    let nl = !p2 & p1 & p0;
    Cls {
        l: a & !p0 & valid,
        n: a & p0,
        other: p1 & !p0 & valid, // codes 2 and 6 — "other" and the apostrophe
        nl,
        sp: p2 & !p1 & !p0,
        ws: (p2 & !p1) | nl, // Space ∪ WsOther ∪ Newline
        apo: p2 & p1 & !p0,
    }
}

/// Class bits of a dense code, for the block-edge carry.
const C_L: u8 = 1;
const C_N: u8 = 2;
const C_O: u8 = 4;
const C_NL: u8 = 8;
const C_SP: u8 = 16;
const C_WS: u8 = 32;

#[inline]
const fn code_bits(code: u8) -> u8 {
    match code {
        0 => C_L,
        1 => C_N,
        2 | 6 => C_O,
        3 => C_NL | C_WS,
        4 => C_SP | C_WS,
        5 => C_WS,
        _ => 0, // cont never reaches an edge (the fill resolves it)
    }
}

/// GPT-2 / byte-level pre-tokenization. `starts` and `flag` are scratch bitmaps
/// (len ≥ `text.len().div_ceil(64)`); byte-exact with `atomsplit::fsm::fsm_byte_level`.
#[must_use]
pub fn bitsplit_byte_level(
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
    assert!(tags.len() >= ntext && starts.len() >= nblk && flag.len() >= nblk && out.len() >= ntext);
    let (mut code, mut prev_cont) = (CODE_CONT, 0u64);

    for bi in 0..nblk {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let last_blk = base + len == ntext;

        let (b, last_code) = build_block::<false>(text, tags, base, len, &LUT, code, false);
        let c = decode(b.p0, b.p1, b.p2, valid);

        let pb = if base == 0 { 0 } else { code_bits(code) };
        let (nb, nb_lead) = if last_blk {
            (0u8, true)
        } else {
            let q = base + len;
            let is_lead = tags[q] != CONT;
            (
                if is_lead {
                    code_bits(LUT[tags[q] as usize])
                } else {
                    code_bits(last_code)
                },
                is_lead,
            )
        };
        let has = |v: u8, s: u8| v & s != 0;
        let p1 = |x: u64, k: bool| (x << 1) | u64::from(k);
        let n1 = |x: u64, k: bool| (x >> 1) | (u64::from(k) << 63);

        let lead = valid & !b.cont;
        let lb = ((lead >> 1) & valid) | (u64::from(nb_lead) << (len - 1));

        // ── every alternative but the contraction is ` ?X+` over a class run, so a token opens at
        // each run start — pushed back one char when a literal space sits in front of it (` ?`).
        let sp_pfx = p1(c.sp, has(pb, C_SP));
        let l_start = c.l & lead & !p1(c.l, has(pb, C_L)) & !sp_pfx;
        let n_start = c.n & lead & !p1(c.n, has(pb, C_N)) & !sp_pfx;
        let o_start = c.other & lead & !p1(c.other, has(pb, C_O)) & !sp_pfx;
        let ws_start = c.ws & lead & !p1(c.ws, has(pb, C_WS));
        // `\s+(?!\S)` hands the run's LAST whitespace char to whatever follows: as a ` ?` prefix if
        // it is a space, else as a token of its own. Either way it opens a token — unless the run
        // ends the input, where plain `\s+` takes the lot. GPT-2 has no `[\r\n]` rule, so unlike
        // cl100k a newline is ordinary whitespace here and can be the stolen char.
        let eof_bit = if last_blk { 1u64 << (len - 1) } else { 0 };
        let (steal, patch) = to_lead(
            c.ws & lb & !eof_bit & !n1(c.ws, has(nb, C_WS)),
            b.cont,
            prev_cont,
        );

        let mut st = (l_start | n_start | o_start | ws_start | steal) & lead;
        if bi == 0 {
            st |= 1;
        }
        starts[bi] = st;
        flag[bi] = st & c.apo; // apostrophes that open a token → contraction escape
        if bi > 0 {
            starts[bi - 1] |= patch;
        }
        code = last_code;
        prev_cont = b.cont;
    }
    emit_contr(text, starts, flag, nblk, ntext, false, out)
}

/// cl100k_base / Llama-3 pre-tokenization. Byte-exact with `atomsplit::fsm::fsm_cl100k`.
#[must_use]
pub fn bitsplit_cl100k(
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
    assert!(tags.len() >= ntext && starts.len() >= nblk && flag.len() >= nblk && out.len() >= ntext);
    let (mut code, mut prev_cont) = (CODE_CONT, 0u64);
    let (mut nl_run, mut dig_run, mut dig_since) = (false, false, 0u32);
    let mut prev_osf = false; // previous block's last byte belonged to a token-opening "other" char
    let mut anl: Option<usize> = None;

    for bi in 0..nblk {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let last_blk = base + len == ntext;

        let (b, last_code) = build_block::<false>(text, tags, base, len, &LUT, code, false);
        let c = decode(b.p0, b.p1, b.p2, valid);

        let pb = if base == 0 { 0 } else { code_bits(code) };
        let (nb, nb_lead) = if last_blk {
            (0u8, true)
        } else {
            let q = base + len;
            let is_lead = tags[q] != CONT;
            (
                if is_lead {
                    code_bits(LUT[tags[q] as usize])
                } else {
                    code_bits(last_code)
                },
                is_lead,
            )
        };
        let has = |v: u8, s: u8| v & s != 0;
        let p1 = |x: u64, k: bool| (x << 1) | u64::from(k);
        let n1 = |x: u64, k: bool| (x >> 1) | (u64::from(k) << 63);

        let lead = valid & !b.cont;
        let lb = ((lead >> 1) & valid) | (u64::from(nb_lead) << (len - 1));
        let eof_bit = if last_blk { 1u64 << (len - 1) } else { 0 };

        // ── run starts ─────────────────────────────────────────────────────────────────────────
        let o_start = c.other
            & lead
            & !p1(c.other, has(pb, C_O))
            & !p1(c.sp, has(pb, C_SP)); // ` ?[^\s\p{L}\p{N}]+`
        let ws_start = c.ws & lead & !p1(c.ws, has(pb, C_WS));
        // `\s+(?!\S)`: the run's last char opens a token — but not a newline, which `\s*[\r\n]+`
        // has already swallowed.
        let (steal, steal_patch) = to_lead(
            c.ws & !c.nl & lb & !eof_bit & !n1(c.ws, has(nb, C_WS)),
            b.cont,
            prev_cont,
        );
        // `\s*[\r\n]+` runs through the run's LAST newline, so a token opens right after it unless
        // a further newline still follows inside the run (backward scan → reversal).
        let after_nl = p1(c.nl, has(pb, C_NL))
            & c.ws
            & lead
            & !fill_to_last(c.nl.reverse_bits(), c.ws.reverse_bits()).reverse_bits();

        // ── `[^\r\n\p{L}\p{N}]?\p{L}+`: the prefix is ANY non-newline non-letter non-digit char,
        // not just a space — so a punctuation char that opens a token is swallowed by a following
        // letter run (`x!abc` → `x`, `!abc`), while one in mid-run is not (`x!!abc` → `x`, `!!`,
        // `abc`), because the greedy other-run already owns it.
        //
        // Stated backward ("my predecessor opened a token") rather than forward ("my successor is a
        // letter"): the forward form needs an `adv`, which silently drops the marker when the two
        // chars straddle a block edge. Smearing `o_start` across its char's bytes makes the test a
        // plain `p1`, and the smear's only cross-block state is a shift carry.
        let mut osf = o_start;
        osf |= (osf << 1) & b.cont;
        osf |= (osf << 2) & b.cont & (b.cont << 1);
        if prev_osf {
            osf |= lead_run(b.cont, valid); // a char whose lead sat in the previous block
        }
        let l_start = c.l
            & lead
            & !p1(c.l, has(pb, C_L))
            & !p1(c.ws & !c.nl, has(pb, C_WS) && !has(pb, C_NL))
            & !p1(osf, prev_osf);

        // ── `\p{N}{1,3}`: a group boundary every 3 chars from the run start.
        let mut m = c.n & lead & !p1(c.n, has(pb, C_N));
        if dig_run && has(pb, C_N) {
            let mut s = lead & lead.wrapping_neg() & c.n;
            for _ in 0..((3 - dig_since % 3) % 3) {
                s = adv(s, b.cont) & c.n & lead;
            }
            m |= s;
        }
        let mut groups = m;
        if c.n & b.cont == 0 {
            let n3 = c.n & (c.n << 1) & (c.n << 2);
            while m != 0 {
                m = (m << 3) & n3;
                groups |= m;
            }
        } else {
            while m != 0 {
                let a = adv(m, b.cont) & c.n & lead;
                let e = adv(adv(a, b.cont) & c.n & lead, b.cont) & c.n & lead;
                if e == 0 {
                    break;
                }
                groups |= e;
                m = e;
            }
        }

        // ── the other-run's `[\r\n]*` tail swallows the newlines directly behind it.
        let nl_m = ((p1(c.other, has(pb, C_O)) & c.nl & lead) as u128) | u128::from(nl_run);
        let nl_e = scanthru(nl_m, c.nl as u128);
        let nl_span = nl_e.wrapping_sub(nl_m);

        // ── the one backward-in-time dependency (see deepseek): a newline arriving now retracts an
        // "after the last newline" start committed for a run that was still open at the last edge.
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
        flag[bi] = st & c.apo;
        if bi > 0 {
            starts[bi - 1] |= steal_patch;
        }

        // ── carries ────────────────────────────────────────────────────────────────────────────
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
            counted % 3
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
    emit_contr(text, starts, flag, nblk, ntext, true, out)
}
