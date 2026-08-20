//! DeepSeek-V3/V4 pre-tokenization: the `Sequence` of `\p{N}{1,3}` → `[一-龥぀-ゟ゠-ヿ]+` →
//! the big regex, all `Isolated`, as one bitstream program. Byte-exact with
//! `atomsplit::fsm::fsm_deepseek`.

use crate::{
    Anl, was, will, Digits,
   AUX_CJK, CODE_CONT, Out, Span, blocks, build_block, emit,
    later_in_run, scanthru, to_lead, trail_run,
};

/// Atom tag → dense 3-bit code. Letter and Mark share a code because deepseek's letter run is
/// `[\p{L}\p{M}]+`; `AlphaSymMark` (0x16) is categorically `\p{S}` so it takes the punct path, and
/// `Zwj` (0x26) matches no alternative at all so it is a gap char.
pub(crate) const LUT: [u8; 64] = {
    let mut t = [6u8; 64]; // gap
    t[0x00] = 0;
    t[0x10] = 0;
    t[0x20] = 0;
    t[0x06] = 0; // Letter (+case) | Mark
    t[0x30] = 0; // Script=Han letter — an ordinary letter to deepseek (its CJK arm is a range)
    t[0x01] = 1;
    t[0x02] = 1;
    t[0x31] = 1; // \p{N} (Han Nl included)
    t[0x07] = 2;
    t[0x08] = 2;
    t[0x09] = 2;
    t[0x0A] = 2;
    t[0x16] = 2;
    t[0x3A] = 2; // \p{P} ∪ \p{S} (Han So included)
    t[0x03] = 3;
    t[0x04] = 4;
    t[0x05] = 5; // Newline | Space | WsOther
    t[0x0F] = CODE_CONT;
    t
};

streams!(
    /// deepseek's class streams. `n_raw`/`lm_raw` are pre-CJK-peel: Split-2 outranks the big regex
    /// and `fsm_deepseek` tests it ahead of the digit arm too, so CJK comes off every other class
    /// first — but `gap` and the `(?!\S)` give-back still ask about the raw classes.
    Cls {
        lead,
        cont,
        cjk,
        cjk_l,
        cjk_p,
        num,
        n_raw,
        lm,
        lm_raw,
        ps,
        ws,
        nl,
        sp,
        gap,
        prefix,
    }
);

/// Cross-block state that is *not* a stream: the three open-run flags and the retractable `anl`.
/// Everything a rule asks about a neighbouring byte now rides in the streams themselves.
#[derive(Default)]
struct Carry {
    aa_run: bool,       // an alt-1 `[A-Za-z]+` run is still open
    nl_run: bool,       // a `[\p{P}\p{S}]+[\r\n]*` newline tail is still open
    dig: Digits,
    anl: Anl,
}

/// Build one block's streams. `code`/`cjk_in` describe the byte just before it, so a block opening
/// mid-char keeps inheriting its lead's class; returns the same pair for the block after.
fn cls(
    text: &[u8],
    tags: &[u8],
    base: usize,
    len: usize,
    code: u8,
    cjk_in: bool,
) -> (Cls, u8, bool) {
    let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
    let (b, last_code) =
        build_block::<{ AUX_CJK }, false>(text, tags, base, len, &LUT, code, cjk_in);
    // planes → classes: 3 extractions instead of 7 one-hot masks (see `LUT`). Only `lm`
    // needs the `valid` mask — past the block end every plane reads 0, i.e. code 0.
    let (pa, pc, pd) = (!b.p2 & !b.p1, !b.p2 & b.p1, b.p2 & !b.p1);
    let (lm_raw, n_raw) = (pa & !b.p0 & valid, pa & b.p0);
    let (ps, nl) = (pc & !b.p0, pc & b.p0);
    let (sp, ws) = (pd & !b.p0, pd | (pc & b.p0));
    let cjk = b.aux;
    let gap = valid & !(n_raw | lm_raw | ps | ws | cjk);
    let c = Cls {
        lead: valid & !b.cont,
        cont: b.cont,
        cjk,
        cjk_l: cjk & lm_raw, // Split-3 re-splits the isolated CJK run into same-kind sub-runs
        cjk_p: cjk & !lm_raw,
        num: n_raw & !cjk,
        n_raw,
        lm: lm_raw & !cjk,
        lm_raw,
        ps: ps & !cjk,
        ws,
        nl,
        sp,
        gap,
        // `[^\r\n\p{L}\p{P}\p{S}]?` — the letter alternative's optional one-char prefix. Digits and
        // CJK are already isolated by Split-1/2, so what remains is non-newline ws + gap chars.
        prefix: (ws & !nl) | gap,
    };
    (c, last_code, cjk >> (len - 1) & 1 != 0)
}

/// Pre-tokenize `text` (well-formed UTF-8) with the DeepSeek grammar: writes token spans into `out`
/// and returns the count. `tags` is `atomsplit::classify`'s output (len ≥ `text.len()`), `starts`
/// is scratch for the token-start bitmap (len ≥ `text.len().div_ceil(64)`).
#[must_use]
pub fn bitsplit_deepseek(text: &[u8], tags: &[u8], starts: &mut [u64], out: &mut [Span]) -> usize {
    let ntext = text.len();
    if ntext == 0 {
        return 0;
    }
    assert!(tags.len() >= ntext && starts.len() >= ntext.div_ceil(64) && out.len() >= ntext);
    let nblk = ntext.div_ceil(64);
    let mut cy = Carry::default();

    blocks(
        ntext,
        &mut *starts,
        None,
        (CODE_CONT, false),
        |base, len, (code, cjk_in)| {
            let (c, last, last_cjk) = cls(text, tags, base, len, code, cjk_in);
            (c, (last, last_cjk))
        },
        |x, starts| {
            let (pv, cur, bk, fw) = (&x.pv, &x.cur, &x.bk, &x.fw);
            let (valid, len) = (x.valid, x.len);
            // a char that is both its own first and last byte is single-byte, i.e. ASCII; an ASCII
            // char in `\p{L}∪\p{M}` is exactly `[A-Za-z]` (ASCII has no marks).
            let ascii = cur.lead & x.lb;
            let aa = cur.lm_raw & ascii;

            // ── run starts. All purely backward-looking, so the carry alone makes them exact. ──
            let lm_start = cur.lm & cur.lead & !bk.lm & !bk.prefix;
            let ws_start = cur.ws & cur.lead & !bk.ws;
            let gap_start = cur.gap & cur.lead & !bk.gap;
            let ps_start = cur.ps & cur.lead & !bk.ps & !bk.sp;
            let cjk_start =
                (cur.cjk_l & cur.lead & !bk.cjk_l) | (cur.cjk_p & cur.lead & !bk.cjk_p);

            // ── Split-1 `\p{N}{1,3}` — a group boundary every 3 chars from the run start. This
            // was `digit_groups` written out by hand, fast path included; it is the shared one now.
            let groups = cy.dig.starts(3, cur.num, cur.lead, cur.cont, was(pv.num));

            // ── whitespace: `\s*[\r\n]+ | \s+(?!\S) | \s+`.
            // (a) the run's first token runs through its LAST newline → a start right after it,
            //     unless a further newline still follows inside the run (a backward scan, hence
            //     the reversal).
            let anl = bk.nl & cur.ws & cur.lead;
            let later_nl =
                later_in_run(cur.nl, cur.ws);
            let after_nl = anl & !later_nl;
            // (b) the run's last char is handed to whatever follows, as its `[^…]?` / ` ?` prefix
            //     — unless the run ends the input or the next piece is Split-1/2-isolated (`(?!\S)`).
            let steal_lb =
                cur.ws & !cur.nl & x.lb & !x.eof & !fw.ws & !(fw.n_raw | fw.cjk);
            let (steal, steal_patch) = to_lead(steal_lb, cur.cont, pv.cont);
            // (c) the same one-char give-back out of a gap run (Control / NumericOther / ZWJ match
            // no alternative, so the run is one piece minus the char a following letter run claims).
            let (gap_steal, gap_patch) = to_lead(cur.gap & x.lb & fw.lm, cur.cont, pv.cont);

            // ── alt-1 `[ascii_punct][A-Za-z]+`: fires only where the scan is actually positioned,
            // i.e. at a punct-run start no space swallowed. Its `[A-Za-z]+` run then has no
            // interior starts and forces one at its end — which the letter rule alone would
            // suppress (`!c` reads `!ab`, `c` after it is a fresh token even though its
            // predecessor is a letter).
            //
            // The one raw-byte peek left: `aa` needs `lb`, which needs the block *after* next, so
            // asking "is the next char an ASCII letter?" stays a byte test rather than a third
            // carry level.
            let nb_aa = !x.last_blk && text[x.base + len].is_ascii_alphabetic();
            let alt1 = ps_start & ascii & ((aa >> 1) | (u64::from(nb_aa) << 63));
            let aa_m = ((alt1 as u128) << 1) | u128::from(cy.aa_run);
            let aa_e = scanthru(aa_m, aa as u128);
            let aa_span = aa_e.wrapping_sub(aa_m);
            // ── a punct run's `[\r\n]*` tail swallows the newlines directly behind it.
            let nl_m = ((bk.ps & cur.nl & cur.lead) as u128) | u128::from(cy.nl_run);
            let nl_e = scanthru(nl_m, cur.nl as u128);
            let nl_span = nl_e.wrapping_sub(nl_m);

            cy.anl.retract(starts, was(pv.ws), cur.ws, cur.nl, valid);

            let mut st = groups
                | lm_start
                | ws_start
                | gap_start
                | ps_start
                | cjk_start
                | after_nl
                | steal
                | gap_steal;
            st &= !(aa_span as u64) & !(nl_span as u64);
            st |= aa_e as u64 | nl_e as u64;

            // ── carries ───────────────────────────────────────────────────────────────────────
            cy.aa_run = aa_e >> 64 != 0;
            cy.nl_run = nl_e >> 64 != 0;
            cy.dig.commit(3, groups, cur.num, cur.lead, valid, len, will(x.nx.num));
            let tws = trail_run(cur.ws, valid, len);
            cy.anl.commit(x.base, after_nl, tws, nl_e as u64, will(x.nx.ws), was(pv.ws));

            Out {
                st,
                patch: steal_patch | gap_patch,
                flag: 0,
            }
        },
    );

    emit(starts, nblk, ntext, out)
}
