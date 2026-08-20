//! **cl100k_base** (tiktoken) / Llama-3 -- and byte-for-byte the regex **GLM-4.6** ships:
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Rule 3's digit cap is the family's only knob, so it is a plain argument: 3 = cl100k / Llama-3 /
//! GLM, 1 = **Qwen** (`\p{N}`), 0 or >= 64 = an unbounded `\p{N}+`.
//!
//! The atom table and the class decode are shared with GPT-2 in [`super::family_gpt`].

use super::family_gpt::cls;
use crate::{
    Anl, CODE_CONT, Digits, Out, Span, blocks, was, will, emit_contr, fill_to_last,
    later_in_run, scanthru, to_lead, trail_run,
};

/// cl100k_base / Llama-3 / GLM-4.6 — rule 3 is `\p{N}{1,3}`.
#[must_use]
pub fn bitsplit_cl100k(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    cl100k(text, tags, starts, flag, out, 3)
}

/// Qwen2 / Qwen3 — cl100k character-for-character except rule 3 is a bare `\p{N}`.
#[must_use]
pub fn bitsplit_qwen(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    cl100k(text, tags, starts, flag, out, 1)
}

fn cl100k(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
    digit_cap: usize,
) -> usize {
    let ntext = text.len();
    if ntext == 0 {
        return 0;
    }
    let nblk = ntext.div_ceil(64);
    assert!(
        tags.len() >= ntext && starts.len() >= nblk && flag.len() >= nblk && out.len() >= ntext
    );
    let mut nl_run = false;
    let mut dig = Digits::default();
    let mut prev_osf = false; // previous block's last byte belonged to a token-opening "other" char
    let mut anl = Anl::default();

    blocks(
        ntext,
        &mut *starts,
        Some(&mut *flag),
        CODE_CONT,
        |base, len, seed| cls(text, tags, base, len, seed),
        |x, starts| {
            let (pv, cur, bk, fw) = (&x.pv, &x.cur, &x.bk, &x.fw);
            let (valid, len, lead) = (x.valid, x.len, x.cur.lead);

            // ── run starts ─────────────────────────────────────────────────────────────────────
            let o_start = cur.other & lead & !bk.other & !bk.sp; // ` ?[^\s\p{L}\p{N}]+`
            let ws_start = cur.ws & lead & !bk.ws;
            // `\s+(?!\S)`: the run's last char opens a token — but not a newline, which
            // `\s*[\r\n]+` has already swallowed.
            let (steal, steal_patch) =
                to_lead(cur.ws & !cur.nl & x.lb & !x.eof & !fw.ws, cur.cont, pv.cont);
            // `\s*[\r\n]+` runs through the run's LAST newline, so a token opens right after it
            // unless a further newline still follows inside the run (backward scan → reversal).
            let after_nl = bk.nl
                & cur.ws
                & lead
                & !later_in_run(cur.nl, cur.ws);

            // ── `[^\r\n\p{L}\p{N}]?\p{L}+`: the prefix is ANY non-newline non-letter non-digit
            // char, not just a space — so a punctuation char that opens a token is swallowed by a
            // following letter run (`x!abc` → `x`, `!abc`), while one in mid-run is not (`x!!abc`
            // → `x`, `!!`, `abc`), because the greedy other-run already owns it.
            //
            // Stated backward ("my predecessor opened a token") rather than forward ("my successor
            // is a letter"): the forward form needs an `adv`, which silently drops the marker when
            // the two chars straddle a block edge. Smearing `o_start` across its char's bytes makes
            // the test a plain shift, and the smear's only cross-block state is a shift carry.
            // `osf`: each start smeared across its char's continuation bytes, so "my predecessor
            // opened a token" is a plain shift. That is `MatchStar` over `cont`, seeded at bit 0 as
            // well when the char's lead sat in the previous block.
            // A start smeared across its char's continuation bytes, so "my predecessor opened a
            // token" is a plain shift. `MatchStar` consumes `c` from the marker, and the marker is a
            // LEAD, so step into the run first; bit 0 is seeded too when the char's lead sat in the
            // previous block.
            let osf = fill_to_last(
                ((o_start << 1) | u64::from(prev_osf)) & cur.cont,
                cur.cont,
            ) | o_start;
            let l_start = cur.l
                & lead
                & !bk.l
                & !(bk.ws & !bk.nl)
                & !((osf << 1) | u64::from(prev_osf));

            // ── rule 3 `\p{N}{1,digit_cap}`
            let groups = dig.starts(digit_cap, cur.n, lead, cur.cont, was(pv.n));

            // ── the other-run's `[\r\n]*` tail swallows the newlines directly behind it.
            let nl_m = ((bk.other & cur.nl & lead) as u128) | u128::from(nl_run);
            let nl_e = scanthru(nl_m, cur.nl as u128);
            let nl_span = nl_e.wrapping_sub(nl_m);

            anl.retract(starts, was(pv.ws), cur.ws, cur.nl, valid);

            let mut st = groups | l_start | o_start | ws_start | after_nl | steal;
            st &= !(nl_span as u64);
            st |= nl_e as u64;

            // ── carries ────────────────────────────────────────────────────────────────────────
            nl_run = nl_e >> 64 != 0;
            dig.commit(digit_cap, groups, cur.n, lead, valid, len, will(x.nx.n));
            let tws = trail_run(cur.ws, valid, len);
            anl.commit(x.base, after_nl, tws, nl_e as u64, will(x.nx.ws), was(pv.ws));
            prev_osf = osf >> (len - 1) & 1 != 0;

            Out {
                st,
                patch: steal_patch,
                flag: cur.apo,
            }
        },
    );
    emit_contr(text, starts, flag, nblk, ntext, true, out)
}
