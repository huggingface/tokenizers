//! **GPT-2 / ByteLevel**:
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Contractions are an ALTERNATIVE here (`don't` -> `don`, `'t`), and case-sensitive -- unlike
//! cl100k's `(?i:)`. They go to `emit_contr`'s scalar escape: a variable-length, case-optional
//! literal alternation that outranks every other arm is miserable in bit algebra and trivial there.

use crate::{
    AUX_NONE, CODE_CONT, CONT, Span, build_block, emit_contr, to_lead,
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
    assert!(
        tags.len() >= ntext && starts.len() >= nblk && flag.len() >= nblk && out.len() >= ntext
    );
    let (mut code, mut prev_cont) = (CODE_CONT, 0u64);

    for bi in 0..nblk {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let last_blk = base + len == ntext;

        let (b, last_code) = build_block::<{ AUX_NONE }, false>(text, tags, base, len, &LUT, code, false);
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

