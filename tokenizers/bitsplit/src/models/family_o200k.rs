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

use crate::classify::{char_len, in_mask, mask};
use crate::{
    AUX_SLASH, CODE_CONT, CONT, Out, Span, blocks, build_block, contr_len, digit_groups,
    fill_to_last, lead_run, letter_match, member, run_end, scanthru, to_lead, trail_run, ws_tail,
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
    prev_osf: bool,
    prev_absorbed: bool, // the block's last byte was eaten by a `[\r\n/]*` tail
    dig_run: bool,
    dig_since: u32,
    anl: Option<usize>,
    sfx_carry: u64,  // bytes of a contraction suffix that spilled past the last block edge
    force: u64,      // ...and the start it opens just past itself, if that landed past the edge
    sfx_end: usize,  // just past the last consumed suffix: the apostrophe THERE is a prefix, not
                     // a second suffix -- `(?i:...)?` applies once (`a's's` is `a's`, `'s`)
    lc_open: bool,   // an `lc` run was still open at the last block edge
    prev_start: Option<usize>,
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

/// `later_in_run`: for every position, does an `l` occur at or after it inside the same letter run
/// (`later[..nblk]`), and does a `cm` occur at or after it inside the same `uc`-run
/// (`later[nblk..]`)?
///
/// These are the only questions in the family that look forward past one block: a letter run can
/// span any number of blocks, so the answer cannot be produced by the forward pass's one-block
/// lookahead. Resolved here once, right to left, as their own streams -- a self-contained kernel
/// with no scalar walk in it, which is what lets the grammar stay pure bit algebra.
pub(crate) fn later_in_run<const AUX: u8, const HAN: bool>(
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
        let rev = |v: u64| v.reverse_bits();
        let mut la = rev(fill_to_last(rev(l), rev(letter)));
        let mut ca = rev(fill_to_last(rev(cm), rev(uc)));
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

/// Run the family grammar. `starts` and `flag` are scratch bitmaps (len ≥ `div_ceil(64)`).
#[must_use]
pub(crate) fn run<const AUX: u8, const CONTR: bool, const DIGITS: usize, const HAN: bool>(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
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
            && flag.len() >= nblk
            && later.len() >= 2 * nblk
            && out.len() >= ntext
    );
    later_in_run::<AUX, HAN>(text, tags, later);
    let mut cy = Carry::default();
    let mut code = CODE_CONT;

    blocks(
        ntext,
        &mut *starts,
        |base, len| {
            let (c, last) = cls::<AUX, HAN>(text, tags, base, len, code);
            code = last;
            c
        },
        |x, starts| {
            let (pv, cur, bk, fw) = (&x.pv, &x.cur, &x.bk, &x.fw);
            let (valid, len, lead) = (x.valid, x.len, x.cur.lead);
            let was = |v: u64| v >> 63 != 0;
            let will = |v: u64| v & 1 != 0;
            let letter = cur.letter;

            // Rule 4's `[\r\n/]*` tail. `/` is in BOTH the `+` body and the tail, so one tail run
            // can collect SEVERAL markers (`!\n/\n`: the `\n` after `!` and the `\n` after `/`).
            // That is exactly what `fill_to_last` is for -- `nl_e - nl_m` would span only from the
            // last one.
            let tail_cls = cur.nl | cur.slash;
            let nl_m64 = ((bk.oth & cur.nl & lead) | u64::from(cy.nl_run)) & tail_cls;
            let nl_e = scanthru(nl_m64 as u128, tail_cls as u128);
            let nl_span = fill_to_last(nl_m64, tail_cls);

            // ── rule 4 ` ?[^\s\p{L}\p{N}]+[\r\n/]*` and rules 5-7 — identical to cl100k.
            // A char absorbed by a `[\r\n/]*` tail is NOT part of the `+` body, so an "other" after
            // it opens a fresh run (`\u{1f600}\r\n/#` is the tail, then `#` starts again).
            let o_prev = cur.oth & !nl_span;
            let o_start = cur.oth
                & lead
                & !nl_span // a char INSIDE the tail never opens a run
                & !((o_prev << 1) | u64::from(was(pv.oth) && !cy.prev_absorbed))
                & !bk.sp;
            let ws_start = cur.ws & lead & !bk.ws;
            let (steal, steal_patch) = to_lead(
                cur.ws & !cur.nl & x.lb & !x.eof & !fw.ws,
                cur.cont,
                pv.cont,
            );
            let after_nl = bk.nl
                & cur.ws
                & lead
                & !fill_to_last(cur.nl.reverse_bits(), cur.ws.reverse_bits()).reverse_bits();

            // ── `[^\r\n\p{L}\p{N}]?` before a letter run: any token-opening non-newline
            // non-letter non-digit char. Smeared across its char's bytes so the test is a plain
            // shift (see cl100k).
            let mut osf = o_start;
            osf |= (osf << 1) & cur.cont;
            osf |= (osf << 2) & cur.cont & (cur.cont << 1);
            if cy.prev_osf {
                osf |= lead_run(cur.cont, valid);
            }
            let l_start = letter
                & lead
                & !bk.letter
                & !(bk.ws & !bk.nl)
                & !((osf << 1) | u64::from(cy.prev_osf));
            // ── the contraction suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)?`, resolved FIRST because its
            // letters belong to the token before it: leaving them in `letter` would make the
            // alternation below start its run on the suffix (`…'s中Ĳa中` would split after `中`).
            let apo_after = if CONTR { cur.apo & lead & bk.letter } else { 0 };
            let mut sfx = cy.sfx_carry;  // a suffix that spilled past the last edge
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
            let case_gate = cur.u & lead & bk.letter;
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
            // `\p{M}` is in BOTH the letter classes and rule 4's `[^\s\p{L}\p{N}]`, and which one
            // wins depends on whether the punctuation before it STARTED the run (`!\u{301}a` is one
            // token, `!!\u{301}a` is two). Adjacency is the gate; the scalar pass resolves it. Real
            // text hits this via emoji + variation selector (U+FE0F is \p{Mn}).
            let mark_adj = (cur.mark & lead & bk.oth) | (cur.oth & lead & bk.mark);

            // ── rule 3 `\p{N}{1,DIGITS}`
            let groups = digit_groups(
                DIGITS,
                cur.n,
                lead,
                cur.cont,
                was(pv.n),
                cy.dig_run,
                cy.dig_since,
            );

            // ── the one backward-in-time rule, exactly as in cl100k/deepseek: a newline arriving
            // now retracts an "after the last newline" start committed for a run still open at the
            // edge.
            if let Some(p) = cy.anl
                && was(pv.ws)
                && cur.ws & 1 != 0
                && cur.nl & lead_run(cur.ws, valid) != 0
            {
                starts[p / 64] &= !(1u64 << (p % 64));
                cy.anl = None;
            }

            let mut st = groups | l_start | lt_in | lt_cut | han_start | o_start | ws_start | after_nl | steal;
            st &= !nl_span;
            st |= nl_e as u64;
            st &= lead;
            st &= !sfx; // nothing inside a contraction suffix opens a token
            st |= sfx_open;
            if x.bi == 0 {
                st |= 1;
            }

            // ── the only escape left: `mark_adj`. `\p{M}` is in BOTH the letter classes and
            // rule 4's `[^\s\p{L}\p{N}]`, and which one wins depends on whether the punctuation
            // before it STARTED the run (`!\u{301}a` is one token, `!!\u{301}a` is two). Rare -- 0%
            // of english/code/russian/chinese blocks, 28% of hindi -- so be blunt: escape every
            // token in the block, plus the one still open from an earlier block (rule 4's ` ?` means
            // it can have started on a space).
            flag[x.bi] = 0;
            if mark_adj != 0 {
                flag[x.bi] = st;
                if let Some(p) = cy.prev_start {
                    flag[p / 64] |= 1u64 << (p % 64);
                }
            }
            if st != 0 {
                cy.prev_start = Some(x.base + 63 - st.leading_zeros() as usize);
            }

            // ── carries
            cy.nl_run = nl_e >> 64 != 0;
            cy.prev_absorbed = nl_span >> (len - 1) & 1 != 0;
            let tn = trail_run(cur.n, valid, len);
            cy.dig_run = tn != 0 && will(x.nx.n);
            cy.dig_since = if !cy.dig_run {
                0
            } else {
                let g = groups & tn;
                let counted = if g == 0 {
                    cy.dig_since + (cur.n & lead & tn).count_ones()
                } else {
                    (cur.n & lead & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
                };
                counted % DIGITS as u32
            };
            let tws = trail_run(cur.ws, valid, len);
            if tws != 0 && will(x.nx.ws) {
                let a = after_nl & tws & !(nl_e as u64);
                if a != 0 {
                    cy.anl = Some(x.base + 63 - a.leading_zeros() as usize);
                } else if !(tws & 1 != 0 && was(pv.ws)) {
                    cy.anl = None;
                }
            } else {
                cy.anl = None;
            }
            cy.prev_osf = osf >> (len - 1) & 1 != 0;

            Out {
                st,
                patch: steal_patch,
            }
        },
    );

    emit::<HAN, CONTR, DIGITS>(text, tags, starts, flag, nblk, ntext, out)
}

// ── the scalar escape ───────────────────────────────────────────────────────────────────────────
// Ported from the FSM this replaces, which is the proven reading of the alternatives. It runs from
// a flagged token start and RESYNCS the moment it lands on an algebra start bit again, so a
// divergent region costs a short scalar walk and nothing downstream.

/// Emit ONE token starting at `i` (letters emit several) and return the cursor past it.
fn step<const HAN: bool, const CONTR: bool, const DIGITS: usize>(
    text: &[u8],
    tags: &[u8],
    i: usize,
    end: usize,
    out: &mut [Span],
    w: &mut usize,
) -> usize {
    // kimi subtracts Han from both letter classes and isolates it in its own arm. Script=Han is
    // refinement 3 of whichever coarse class it lands in (0x30 letter / 0x31 Nl / 0x3A So) and
    // nothing else uses that nibble, so this is one mask instead of a per-byte range search.
    let han = |p: usize| HAN && p < end && tags[p] & 0xF0 == 0x30;
    let is_lm = |p: usize| p < end && member(tags[p]) && !han(p);
    let letter_end = |mut p: usize| {
        while p < end && (tags[p] == CONT || (member(tags[p]) && !han(p))) {
            p += 1;
        }
        p
    };
    let other = |sp0: usize| {
        let mut p = run_end(tags, sp0, end, mask::NOT_WS_L_N);
        if p > sp0 {
            // `/` is in the tail for o200k and tekken; kimi's tail is a plain `[\r\n]*`
            while p < end && (tags[p] == crate::NLN || (!HAN && text[p] == b'/')) {
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
            let te = if CONTR && e == re {
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

    // `[\p{Han}]+` — kimi's alternative 1, ahead of everything else
    if HAN && han(i) {
        let mut p = i;
        while p < end && (tags[p] == CONT || han(p)) {
            if tags[p] != CONT && !han(p) {
                break;
            }
            p += 1;
        }
        emit1(i, p, out, w);
        return p;
    }

    let b = text[i];
    match tags[i] & 0x0F {
        crate::NW | crate::NO => {
            let (mut p, mut cnt) = (i, 0usize);
            while p < end && cnt < DIGITS && in_mask(tags[p], mask::NUMBER) {
                p += char_len(text[p]);
                cnt += 1;
            }
            emit1(i, p, out, w);
            p
        }
        crate::LET | crate::MRK => {
            if member(tags[i]) && !han(i) {
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
fn emit<const HAN: bool, const CONTR: bool, const DIGITS: usize>(
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
                    c = step::<HAN, CONTR, DIGITS>(text, tags, c, n, out, &mut w);
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
