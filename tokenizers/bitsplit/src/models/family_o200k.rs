//! The **o200k regex family** — one implementation, three regexes.
//!
//! o200k_base / GPT-4o (byte-for-byte what **Llama-4, gpt-oss and MiniMax-M2** ship), Mistral
//! `tekken`, and `kimi-k2` are the same grammar with three knobs:
//!
//! | knob | o200k | tekken | kimi |
//! |---|---|---|---|
//! | `CONTR` — letter tokens take a `(?i:'s\|'t\|…)` suffix | yes | no | yes |
//! | `DIGITS` — rule-3 cap, `\p{N}{1,DIGITS}` | 3 | 1 | 3 |
//! | `AUX` — rule-4 tail, and kimi's leading `[\p{Han}]+` arm | `[\r\n/]*` | `[\r\n/]*` | `[\r\n]*` |
//!
//! These were three files. The bitstream half was 97% identical between o200k and tekken and 91%
//! against kimi, and the atom table, `decode` and the whole scalar escape (`step`, `emit`) were
//! byte-identical — so a fix had to be applied three times to be a fix at all. What differs between
//! the three is named in that table and nowhere else.
//!
//! Two things are worth knowing before reading the algebra:
//!
//!  1. **The case split is a scalar escape, on purpose.** `[UC]*[LC]+ | [UC]+[LC]*` has no local
//!     form: `中Qz` is ONE token (a U after a C is not a boundary) but `ʰABC` is two (it is, when no
//!     L follows) — the difference is whether an L appears LATER in the run, so no `bk` decides it.
//!     The bit half instead computes a cheap gate: a letter token is escaped only if the block holds
//!     an interior upper or a trailing apostrophe, so all-lowercase and Capitalised text never pays.
//!  2. **The contraction is a SUFFIX here, not an alternative.** cl100k emits `'t` as its own token;
//!     o200k glues it onto the letter token before it. So this uses `emit_contr_suffix`, which
//!     *extends* the open token instead of opening one — the mirror image of `emit_contr`.

use crate::{
    AUX_SLASH, Anl, CODE_CONT, CONT, Digits, Out, Span, blocks, build_block, contr_len,
    fill_to_last, later_in_run, lead_run, scanthru, to_lead, trail_run, was, will,
};

/// Atom tag → dense 4-bit code. Unlike cl100k, `\p{M}` IS a letter here (both alt classes list
/// `\p{M}`) — but only a true mark: `AlphaSymMark` (0x16, categorically `\p{S}`) and `Zwj` (0x26,
/// `\p{Cf}`) stay "other", which is what keeps `[\p{L}\p{M}]+` off them.
/// `HAN` routes Script=Han to code 11 — kimi's alternative 1, which outranks every other arm.
/// Without it Han is an ordinary caseless letter / number / symbol, which is what o200k and tekken
/// see. This is the whole of kimi's divergence: three table rows, not a second code path.
const fn lut<const HAN: bool>() -> [u8; 64] {
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
    t[0x30] = if HAN { 11 } else { 2 }; // Script=Han letter
    t[0x31] = if HAN { 11 } else { 3 }; // Script=Han \p{N} (Nl)
    t[0x3A] = if HAN { 11 } else { 8 }; // Script=Han \p{S} (So)
    t[0x0F] = CODE_CONT;
    t
}

/// Atom -> code for [`seed_at`]. The Han row differs between the two tables, but a seed only ever
/// reproduces a class the fill already assigned, and both tables agree on every class boundary that
/// a continuation byte can inherit.
const LUT_SEED: [u8; 64] = lut::<false>();

streams!(
    /// `letter` is the union of the alt classes. Han needs no subtracting: with `HAN` it has its
    /// own code, so it is not in `u`/`l`/`c` to begin with (kimi's classes are literally
    /// `[…&&[^\p{Han}]]`). `han` is that code; `slash` is the only remaining AUX stream.
    Cls { lead, cont, u, l, c, mark, n, nl, sp, ws, oth, apo, letter, slash, han }
);

/// Cross-block state that is not a stream.
#[derive(Default)]
struct Carry {
    nl_run: bool,
    oth_edge: bool,      // rule 4's effective `oth` at the previous block's last byte
    letter_edge: bool,   // ...and the adjusted `letter` there
    mark_oth_open: bool, // a mark stretch rule 4 took was still running at the last edge
    prev_osf: bool,
    prev_absorbed: bool, // the block's last byte was eaten by a `[\r\n/]*` tail
    dig: Digits,
    anl: Anl,
    sfx_carry: u64, // bytes of a contraction suffix that spilled past the last block edge
    force: u64,     // ...and the start it opens just past itself, if that landed past the edge
    sfx_end: usize, // just past the last consumed suffix: the apostrophe THERE is a prefix, not
    // a second suffix -- `(?i:...)?` applies once (`a's's` is `a's`, `'s`)
    lc_open: bool, // an `lc` run was still open at the last block edge
}

/// Build one block's streams; returns the fill seed and the Han carry for the block after.
fn cls<const AUX: u8, const HAN: bool>(
    text: &[u8],
    tags: &[u8],
    base: usize,
    len: usize,
    code: u8,
) -> (Cls, u8) {
    let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
    let (b, last_code) =
        build_block::<AUX, true>(text, tags, base, len, &lut::<HAN>(), code, false);
    let (p0, p1, p2, p3) = (b.p0, b.p1, b.p2, b.p3);
    let low = !p3;
    let a = low & !p2;
    let w = low & p2;
    let u = a & !p1 & !p0 & valid; // code 0 — past the block end every plane reads 0, hence `valid`
    let l = a & !p1 & p0;
    let c = a & p1 & !p0;
    let mark = p3 & p1 & !p0; // code 10
    let han = p3 & p1 & p0 & valid; // code 11 — kimi only; zero for the other two
    let cls = Cls {
        lead: valid & !b.cont,
        cont: b.cont,
        u,
        l,
        c,
        mark,
        n: a & p1 & p0,
        nl: w & !p1 & !p0,
        sp: w & !p1 & p0,
        ws: w & !(p1 & p0), // codes 4,5,6 — cont (7) excluded
        oth: p3 & !p1,      // codes 8,9 — the apostrophe is "other" for run purposes
        apo: p3 & p0 & !p1, // code 9
        letter: u | l | c | mark,
        slash: if AUX == AUX_SLASH { b.aux } else { 0 },
        han,
    };
    (cls, last_code)
}

/// The fill seed for a block, recovered locally: only a block that opens mid-char needs one, and
/// then the char's lead is at most three bytes back. That is what lets the backward pass below
/// build each block on its own, with no forward dependency.
fn seed_at(tags: &[u8], base: usize) -> u8 {
    if base == 0 {
        return CODE_CONT;
    }
    let mut p = base - 1;
    while p > 0 && tags[p] == CONT {
        p -= 1;
    }
    LUT_SEED[tags[p] as usize]
}

/// `later_streams`: for every position, does an `l` occur at or after it inside the same letter run
/// (`later[..nblk]`), and does a `cm` occur at or after it inside the same `uc`-run
/// (`later[nblk..]`)?
///
/// These are the only questions in the family that look forward past one block: a letter run can
/// span any number of blocks, so the answer cannot be produced by the forward pass's one-block
/// lookahead. Resolved here once, right to left, as their own streams -- a self-contained kernel
/// with no scalar walk in it, which is what lets the grammar stay pure bit algebra.
pub(crate) fn later_streams<const AUX: u8, const HAN: bool>(
    text: &[u8],
    tags: &[u8],
    later: &mut [u64],
) {
    let ntext = text.len();
    let nblk = ntext.div_ceil(64);
    let (mut l_carry, mut cm_carry) = (false, false);
    for bi in (0..nblk).rev() {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let (c, _) = cls::<AUX, HAN>(text, tags, base, len, seed_at(tags, base));
        let (letter, l) = (c.letter, c.l);
        let uc = letter & !l;
        let cm = (c.c | c.mark) & letter;
        let mut la = later_in_run(l, letter);
        let mut ca = later_in_run(cm, uc);
        // a run leaving this block to the right inherits the answer from the block after it
        if l_carry && letter >> (len - 1) & 1 != 0 {
            la |= trail_run(letter, valid, len);
        }
        if cm_carry && uc >> (len - 1) & 1 != 0 {
            ca |= trail_run(uc, valid, len);
        }
        later[bi] = la;
        later[nblk + bi] = ca;
        l_carry = letter & 1 != 0 && la & 1 != 0;
        cm_carry = uc & 1 != 0 && ca & 1 != 0;
    }
}

/// Run the family grammar. `starts` is a scratch bitmap (len ≥ `div_ceil(64)`), `later` twice
/// that; `_flag` is unused here — the scalar escape it fed is gone — and kept only so the family
/// keeps one shape with its siblings.
#[must_use]
pub(crate) fn run<const AUX: u8, const CONTR: bool, const DIGITS: usize, const HAN: bool>(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    _flag: &mut [u64],
    later: &mut [u64],
    out: &mut [Span],
) -> usize {
    let ntext = text.len();
    if ntext == 0 {
        return 0;
    }
    let nblk = ntext.div_ceil(64);
    assert!(
        tags.len() >= ntext
            && starts.len() >= nblk
            && later.len() >= 2 * nblk
            && out.len() >= ntext
    );
    later_streams::<AUX, HAN>(text, tags, later);
    let mut cy = Carry::default();

    blocks(
        ntext,
        &mut *starts,
        None,
        CODE_CONT,
        |base, len, seed| cls::<AUX, HAN>(text, tags, base, len, seed),
        |x, starts| {
            let (pv, cur, bk, fw) = (&x.pv, &x.cur, &x.bk, &x.fw);
            let (valid, len, lead) = (x.valid, x.len, x.cur.lead);

            // Rule 4's `[\r\n/]*` tail. `/` is in BOTH the `+` body and the tail, so one tail run
            // can collect SEVERAL markers (`!\n/\n`: the `\n` after `!` and the `\n` after `/`).
            // That is exactly what `fill_to_last` is for -- `nl_e - nl_m` would span only from the
            // last one.
            let tail_cls = cur.nl | cur.slash;

            // ── rule 4 ` ?[^\s\p{L}\p{N}]+[\r\n/]*` and rules 5-7 — identical to cl100k.
            //
            // `oth` is a fixpoint because `\p{M}` is in BOTH the letter classes and rule 4's, so a
            // mark rule 4 keeps joins its run, which can suppress a later `o_start` and demote
            // further marks. Monotone (`oth` only grows) and bounded by the marks in the block; with
            // no marks it settles on the first pass.
            let mut oth = cur.oth;
            let mut o_start;
            let mut osf;
            let mut mark_oth;
            let (mut nl_e, mut nl_span);
            loop {
                let bk_oth = (oth << 1) | u64::from(cy.oth_edge);
                // The carried marker must NOT be masked with `tail_cls`: the carry says a tail was
                // in flight at the edge, so if bit 0 is not in the class the tail LANDED there and
                // that position opens a token. Masking it dropped the start whenever a tail ended
                // exactly on a block edge (`…!\n` | ` \r`). `nl_span` still masks, because a landing
                // is not *inside* the tail.
                let nl_m64 = (bk_oth & cur.nl & lead) | u64::from(cy.nl_run);
                nl_e = scanthru(nl_m64 as u128, tail_cls as u128);
                nl_span = fill_to_last(nl_m64 & tail_cls, tail_cls);

                // A char absorbed by a `[\r\n/]*` tail is NOT part of the `+` body, so an "other"
                // after it opens a fresh run (`\u{1f600}\r\n/#` is the tail, then `#` starts again).
                let o_prev = oth & !nl_span;
                o_start = oth
                    & lead
                    & !nl_span // a char INSIDE the tail never opens a run
                    & !((o_prev << 1) | u64::from(cy.oth_edge && !cy.prev_absorbed))
                    & !bk.sp;

                // `[^\r\n\p{L}\p{N}]?` before a letter run: any token-opening non-newline
                // non-letter non-digit char, smeared across its char's continuation bytes so the
                // test downstream is a plain shift (see cl100k).
                osf = fill_to_last(
                    ((o_start << 1) | u64::from(cy.prev_osf)) & cur.cont,
                    cur.cont,
                ) | o_start;

                // Which arm claims a `\p{M}`? Two observations settle it:
                //
                //  * only a mark INSIDE rule 4's `+` body is contested at all. One after a letter is
                //    just part of the letter run (`a\u{301}`) -- `[LC]+` is greedy and marks are in
                //    `LC`. One after a char already absorbed by a `[\r\n/]*` tail is not contested
                //    either, because that run has closed (`!\n/\u{301}`), hence `o_prev`.
                //  * alt-1 takes at most ONE char as its `[^\r\n\p{L}\p{N}]?` prefix, so it can only
                //    claim a stretch sitting immediately after the run's FIRST char: `!\u{301}a` is
                //    one token, `!!\u{301}a` is `!!\u{301}` then `a`.
                let contested = fill_to_last(
                    ((o_prev << 1) | u64::from(cy.oth_edge && !cy.prev_absorbed)) & cur.mark & lead,
                    cur.mark,
                );
                let claimed = fill_to_last(
                    ((osf << 1) | u64::from(cy.prev_osf)) & cur.mark & lead,
                    cur.mark,
                );
                mark_oth = contested & !claimed;
                if cy.mark_oth_open {
                    // a stretch rule 4 already took, still running at the last edge: its lead is in
                    // the previous block, so `contested`'s `& lead` cannot see it
                    mark_oth |= lead_run(cur.mark, valid);
                }
                let grown = cur.oth | mark_oth;
                if grown == oth {
                    break;
                }
                oth = grown;
            }
            let letter = cur.letter & !mark_oth;
            let bk_letter = (letter << 1) | u64::from(cy.letter_edge);

            let ws_start = cur.ws & lead & !bk.ws;
            let (steal, steal_patch) =
                to_lead(cur.ws & !cur.nl & x.lb & !x.eof & !fw.ws, cur.cont, pv.cont);
            let after_nl = bk.nl & cur.ws & lead & !later_in_run(cur.nl, cur.ws);

            let l_start = letter
                & lead
                & !bk_letter
                & !(bk.ws & !bk.nl)
                & !((osf << 1) | u64::from(cy.prev_osf));
            // ── the contraction suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)?`, resolved FIRST because its
            // letters belong to the token before it: leaving them in `letter` would make the
            // alternation below start its run on the suffix (`…'s中Ĳa中` would split after `中`).
            let apo_after = if CONTR { cur.apo & lead & bk_letter } else { 0 };
            let mut sfx = cy.sfx_carry; // a suffix that spilled past the last edge
            let mut sfx_open = cy.force; // ...and the start it opens just past itself
            cy.sfx_carry = 0;
            cy.force = 0;
            if CONTR {
                let mut a = apo_after;
                while a != 0 {
                    let j = a.trailing_zeros() as usize;
                    a &= a - 1;
                    let p = x.base + j;
                    if p == cy.sfx_end {
                        continue; // its predecessor is inside a suffix: `?` applies once
                    }
                    let k = contr_len(text, p, true);
                    if k == 0 {
                        continue;
                    }
                    cy.sfx_end = p + k;
                    let hi = j + k;
                    let inb = if hi >= 64 { !0u64 } else { (1u64 << hi) - 1 };
                    sfx |= inb & !((1u64 << j) - 1);
                    if hi > 64 {
                        cy.sfx_carry |= (1u64 << (hi - 64)) - 1;
                    }
                    // the suffix ENDS its token, so the next char opens one -- nothing else would,
                    // since it sits mid-letter-run (`a'sa` is `a's`, `a`).
                    if p + k < ntext {
                        if hi < 64 {
                            sfx_open |= 1u64 << hi;
                        } else {
                            cy.force |= 1u64 << (hi - 64);
                        }
                    }
                }
            }

            // ── the letter alternation `[UC]*[LC]+ | [UC]+[LC]*`, in bits.
            //
            // Ordered alternation with greedy quantifiers means every token is ONE maximal `uc`-run
            // followed by ONE maximal `lc`-run, so a marker at a letter-run start walks the run in
            // pairs -- the same marker iteration rule 3 uses. `c` and `mark` are in BOTH classes,
            // which is the only subtlety: `中Qz` is one token while `aB` is two.
            let lalt = letter & !sfx; // the suffix belongs to the token before it
            // The gate: a letter run splits only where an uppercase sits after a letter. Without one
            // the run is a single token whatever the classes are, so all-lowercase and Capitalised
            // text -- most text -- skips the algebra below entirely.
            let case_gate = cur.u & lead & bk_letter;
            let uc = lalt & !cur.l; // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
            let lc = lalt & !cur.u; // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
            let cm = (cur.c | cur.mark) & lalt; // in both classes
            // A token is one maximal `uc`-run followed by one maximal `lc`-run, so the phase inside
            // a letter run flips at its first strictly-lowercase char -- a caseless char stays in the
            // `uc` phase, because `uc*` is greedy. An uppercase after an `l` therefore opens a new
            // token, and every `l` marker scanned through `lc` lands on exactly that uppercase. So
            // the whole case split is one `ScanThru`: `中Qz` stays whole (its `l` is last, and lands
            // past the run) while `aB` and `AbCd` split. The carry is one bit -- an `lc` run still
            // open at the edge -- and a marker resumed at bit 0 behaves like any other, so there is
            // no phase to track.
            let mm = ((cur.l & lalt) as u128) | u128::from(cy.lc_open);
            let f = scanthru(mm, lc as u128);
            let lt_in = (f as u64) & lalt;
            cy.lc_open = f >> 64 != 0;

            // The one backtrack: alt-1 needs an `lc` AFTER `uc*`, so a `uc`-run that ENDS the letter
            // run cuts after its last `cm` (`ʰABC` -> `ʰ`,`ABC`). `中Qz` is untouched because an `l`
            // follows. "is there an X later in this run" is `fill_to_last` on reversed streams.
            // `later_in_run` answered both "is there an X at or after me in this run?" questions
            // for the whole text, so the cut is exact even where the run spans blocks.
            let lt_cut = if case_gate == 0 {
                0
            } else {
                uc & lead
                    & ((cm << 1) | u64::from(was(pv.c) || was(pv.mark)))
                    & !later[nblk + x.bi]
                    & !later[x.bi]
            };

            // ── kimi's `[\p{Han}]+`, ahead of everything else. Zero for the other two.
            let han_start = cur.han & lead & !bk.han;

            // ── the escape gate (see the header). An interior upper, or an apostrophe closing the
            // run, means some letter token in this block needs the scalar case/contraction pass.

            // ── rule 3 `\p{N}{1,DIGITS}`
            let groups = cy.dig.starts(DIGITS, cur.n, lead, cur.cont, was(pv.n));

            cy.anl.retract(starts, was(pv.ws), cur.ws, cur.nl, valid);

            let mut st = groups
                | l_start
                | lt_in
                | lt_cut
                | han_start
                | o_start
                | ws_start
                | after_nl
                | steal;
            st &= !nl_span;
            st |= nl_e as u64;
            st &= !sfx; // nothing inside a contraction suffix opens a token
            st |= sfx_open;

            // ── carries
            cy.nl_run = nl_e >> 64 != 0;
            cy.prev_absorbed = nl_span >> (len - 1) & 1 != 0;
            cy.dig
                .commit(DIGITS, groups, cur.n, lead, valid, len, will(x.nx.n));
            let tws = trail_run(cur.ws, valid, len);
            cy.anl.commit(
                x.base,
                after_nl,
                tws,
                nl_e as u64,
                will(x.nx.ws),
                was(pv.ws),
            );
            cy.prev_osf = osf >> (len - 1) & 1 != 0;
            cy.oth_edge = oth >> (len - 1) & 1 != 0;
            cy.letter_edge = letter >> (len - 1) & 1 != 0;
            cy.mark_oth_open = mark_oth >> (len - 1) & 1 != 0;

            Out {
                st,
                patch: steal_patch,
                flag: 0,
            }
        },
    );

    crate::emit(starts, &[], nblk, ntext, out)
}
