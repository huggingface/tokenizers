//! **GPT-2 / ByteLevel**:
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Contractions are an ALTERNATIVE here (`don't` -> `don`, `'t`), and case-sensitive -- unlike
//! cl100k's `(?i:)`. They go to `emit_contr`'s scalar escape: a variable-length, case-optional
//! literal alternation that outranks every other arm is miserable in bit algebra and trivial there.

use crate::{AUX_NONE, CODE_CONT, Out, Span, blocks, build_block, emit_contr, to_lead};

/// Atom tag → dense 3-bit code, shared by both grammars. Unlike deepseek's table, `Mark` is NOT a
/// letter here (`\p{L}` excludes it, so it belongs to the "other" class), and `Apostrophe` gets its
/// own code so the contraction escape can be flagged with one AND — it is still "other" for every
/// run rule, which `cls` restores.
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

streams!(
    /// One block's class streams. `other` folds the apostrophe code back in; only `l` needs the
    /// `valid` mask, since past the block end every plane reads 0, i.e. code 0.
    Cls { lead, cont, l, n, other, sp, ws, apo }
);

/// Build one block's streams; returns the fill seed for the block after.
fn cls(text: &[u8], tags: &[u8], base: usize, len: usize, code: u8) -> (Cls, u8) {
    let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
    let (b, last_code) =
        build_block::<{ AUX_NONE }, false>(text, tags, base, len, &LUT, code, false);
    let (p0, p1, p2) = (b.p0, b.p1, b.p2);
    let a = !p2 & !p1;
    let nl = !p2 & p1 & p0;
    let c = Cls {
        lead: valid & !b.cont,
        cont: b.cont,
        l: a & !p0 & valid,
        n: a & p0,
        other: p1 & !p0 & valid, // codes 2 and 6 — "other" and the apostrophe
        sp: p2 & !p1 & !p0,
        ws: (p2 & !p1) | nl, // Space ∪ WsOther ∪ Newline
        apo: p2 & p1 & !p0,
    };
    (c, last_code)
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
    let mut code = CODE_CONT;
    blocks(
        ntext,
        &mut *starts,
        Some(&mut *flag),
        |base, len| {
            let (c, last) = cls(text, tags, base, len, code);
            code = last;
            c
        },
        |x, _| {
            let (cur, bk) = (&x.cur, &x.bk);
            // `\s+(?!\S)` hands the run's LAST whitespace char to whatever follows: as a ` ?`
            // prefix if it is a space, else as a token of its own. Either way it opens a token —
            // unless the run ends the input, where plain `\s+` takes the lot. GPT-2 has no
            // `[\r\n]` rule, so unlike cl100k a newline is ordinary whitespace here and can be
            // the stolen char.
            let (steal, patch) = to_lead(cur.ws & x.lb & !x.eof & !x.fw.ws, cur.cont, x.pv.cont);
            Out {
                // every alternative but the contraction is ` ?X+` over a class run, so a token
                // opens at each run start — pushed back one char when a literal space sits in
                // front of it (` ?`).
                st: (cur.l & !bk.l & !bk.sp)
                    | (cur.n & !bk.n & !bk.sp)
                    | (cur.other & !bk.other & !bk.sp)
                    | (cur.ws & !bk.ws)
                    | steal,
                patch,
                flag: cur.apo, // apostrophes that open a token → contraction escape
            }
        },
    );
    emit_contr(text, starts, flag, nblk, ntext, false, out)
}
