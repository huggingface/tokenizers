//! The atom table and class streams shared by **GPT-2 / ByteLevel** and **cl100k_base**.
//!
//! Both grammars fold the 16 atoms into the same dense 3-bit code and decode the same seven class
//! streams — the tables and the decode were byte-identical in both files. Their *rules* genuinely
//! differ (cl100k has `\s*[\r\n]+` and a `[^\r\n\p{L}\p{N}]?` letter prefix; GPT-2 has neither, and
//! treats a newline as ordinary whitespace), so the grammars stay in their own modules.

use crate::{AUX_NONE, CODE_CONT, build_block};

/// Atom tag → dense 3-bit code. Unlike deepseek's table, `Mark` is NOT a letter here (`\p{L}`
/// excludes it, so it belongs to the "other" class), and `Apostrophe` gets its own code so the
/// contraction escape can be flagged with one AND — it is still "other" for every run rule, which
/// `cls` restores.
pub(crate) const LUT: [u8; 64] = {
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
    Cls { lead, cont, l, n, other, nl, sp, ws, apo }
);

/// Build one block's streams; returns the fill seed for the block after.
pub(crate) fn cls(text: &[u8], tags: &[u8], base: usize, len: usize, code: u8) -> (Cls, u8) {
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
        nl,
        sp: p2 & !p1 & !p0,
        ws: (p2 & !p1) | nl, // Space ∪ WsOther ∪ Newline
        apo: p2 & p1 & !p0,
    };
    (c, last_code)
}
