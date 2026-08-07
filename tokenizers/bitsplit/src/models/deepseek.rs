//! DeepSeek-V3/V4 pre-tokenization: the `Sequence` of `\p{N}{1,3}` → `[一-龥぀-ゟ゠-ヿ]+` →
//! the big regex, all `Isolated`, as one bitstream program. Byte-exact with
//! `atomsplit::fsm::fsm_deepseek`.

use crate::{
    AUX_CJK, CODE_CONT, CONT, Span, adv, build_block, emit, fill_to_last, is_cjk_at, lead_run, scanthru,
    to_lead, trail_run,
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
    t[0x01] = 1;
    t[0x02] = 1; // \p{N}
    t[0x07] = 2;
    t[0x08] = 2;
    t[0x09] = 2;
    t[0x0A] = 2;
    t[0x16] = 2; // \p{P} ∪ \p{S}
    t[0x03] = 3;
    t[0x04] = 4;
    t[0x05] = 5; // Newline | Space | WsOther
    t[0x0F] = CODE_CONT;
    t
};

const S_N: u16 = 1 << 0;
const S_LM: u16 = 1 << 1;
const S_PS: u16 = 1 << 2;
const S_WS: u16 = 1 << 3;
const S_NL: u16 = 1 << 4;
const S_SP: u16 = 1 << 5;
const S_CJK: u16 = 1 << 6;
const S_ANY: u16 = S_N | S_LM | S_PS | S_WS | S_CJK;

/// Dense code → class bits, for the block-edge carries.
const fn code_bits(code: u8) -> u16 {
    match code {
        0 => S_LM,
        1 => S_N,
        2 => S_PS,
        3 => S_WS | S_NL,
        4 => S_WS | S_SP,
        5 => S_WS,
        _ => 0, // gap | cont
    }
}

/// Filled class bits of the char starting at `p` (a lead byte).
fn bits_at(text: &[u8], tags: &[u8], p: usize) -> u16 {
    code_bits(LUT[tags[p] as usize]) | if is_cjk_at(text, p) { S_CJK } else { 0 }
}

/// Cross-block state: what the paper resolves with selective recomputation across SMs.
struct Carry {
    code: u8,           // filled dense code of the previous block's last byte (seeds the fill)
    cjk: bool,          // ...and whether its char is in the Split-2 CJK range
    cont: u64,          // the previous block's `cont` stream (for `to_lead` underflow)
    aa_run: bool,       // an alt-1 `[A-Za-z]+` run is still open
    nl_run: bool,       // a `[\p{P}\p{S}]+[\r\n]*` newline tail is still open
    dig_run: bool,      // inside a \p{N} run
    dig_since: u32,     // chars already consumed of the current \p{N}{1,3} group
    anl: Option<usize>, // a committed "start after the last newline" a later newline may retract
}

impl Default for Carry {
    fn default() -> Self {
        // the cont code as the seed: a text opening with a stray continuation byte then classifies
        // as gap under both builders instead of as `Letter` (code 0).
        Self {
            code: CODE_CONT,
            cjk: false,
            cont: 0,
            aa_run: false,
            nl_run: false,
            dig_run: false,
            dig_since: 0,
            anl: None,
        }
    }
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

    for bi in 0..nblk {
        let base = bi * 64;
        let len = (ntext - base).min(64);
        let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
        let last_blk = base + len == ntext;

        let (b, last_code) = build_block::<{ AUX_CJK }, false>(text, tags, base, len, &LUT, cy.code, cy.cjk);
        // planes → classes: 3 extractions instead of 7 one-hot masks (see `LUT`). Only `lm`
        // needs the `valid` mask — past the block end every plane reads 0, i.e. code 0.
        let (pa, pc, pd) = (!b.p2 & !b.p1, !b.p2 & b.p1, b.p2 & !b.p1);
        let (s_lm, s_n) = (pa & !b.p0 & valid, pa & b.p0);
        let (s_ps, s_nl) = (pc & !b.p0, pc & b.p0);
        let (s_sp, s_ws) = (pd & !b.p0, pd | (pc & b.p0));
        let last_cjk = b.aux >> (len - 1) & 1 != 0;
        let last_bits = code_bits(last_code) | if last_cjk { S_CJK } else { 0 };

        // ── edges. The byte before the block is carried; the one after is peeked. One char of
        // lookahead is all the grammar needs outside whitespace runs, so peeking it keeps every
        // "next char is X" rule exact right at the block boundary — no recomputation needed.
        let pb = if base == 0 {
            0
        } else {
            code_bits(cy.code) | if cy.cjk { S_CJK } else { 0 }
        };
        let (nb, nb_lead, nb_aa) = if last_blk {
            (0u16, true, false)
        } else {
            let q = base + len;
            let is_lead = tags[q] != CONT;
            let bits = if is_lead {
                bits_at(text, tags, q)
            } else {
                last_bits
            };
            (bits, is_lead, text[q].is_ascii_alphabetic())
        };
        let has = |v: u16, s: u16| v & s != 0;
        // Split precedence: CJK (Split-2) outranks the big regex, and `fsm_deepseek` tests it ahead
        // of the digit arm too — so peel CJK off every other class first.
        let cjk = b.aux;
        let num = s_n & !cjk;
        let lm = s_lm & !cjk;
        let ps = s_ps & !cjk;
        let gap = valid & !(s_n | s_lm | s_ps | s_ws | cjk);
        let cjk_l = cjk & s_lm; // Split-3 re-splits the isolated CJK run into same-kind sub-runs
        let cjk_p = cjk & !s_lm;
        // `[^\r\n\p{L}\p{P}\p{S}]?` — the letter alternative's optional one-char prefix. Digits and
        // CJK are already isolated by Split-1/2, so what remains is non-newline ws + gap chars.
        let prefix = (s_ws & !s_nl) | gap;

        // previous-byte / next-byte membership. Streams are filled, so at a lead `p1` reads the
        // previous *char*'s class and at a last byte `n1` reads the next char's.
        let p1 = |x: u64, c: bool| (x << 1) | u64::from(c);
        let n1 = |x: u64, c: bool| (x >> 1) | (u64::from(c) << 63);
        let pb_gap = base != 0 && !has(pb, S_ANY);
        let pb_cjk = has(pb, S_CJK);

        let lead = valid & !b.cont;
        let lb = ((lead >> 1) & valid) | (u64::from(nb_lead) << (len - 1));
        // a char that is both its own first and last byte is single-byte, i.e. ASCII; an ASCII char
        // in `\p{L}∪\p{M}` is exactly `[A-Za-z]` (ASCII has no marks). Both streams for 2 ops.
        let ascii = lead & lb;
        let aa = s_lm & ascii;

        // ── run starts. All purely backward-looking, so the carry alone makes them exact. ───────
        let n_start = num & lead & !p1(num, has(pb, S_N) && !pb_cjk);
        let lm_start = lm
            & lead
            & !p1(lm, has(pb, S_LM) && !pb_cjk)
            & !p1(prefix, (has(pb, S_WS) && !has(pb, S_NL)) || pb_gap);
        let ws_start = s_ws & lead & !p1(s_ws, has(pb, S_WS));
        let gap_start = gap & lead & !p1(gap, pb_gap);
        let ps_start = ps & lead & !p1(ps, has(pb, S_PS) && !pb_cjk) & !p1(s_sp, has(pb, S_SP));
        let cjk_start = (cjk_l & lead & !p1(cjk_l, pb_cjk && has(pb, S_LM)))
            | (cjk_p & lead & !p1(cjk_p, pb_cjk && !has(pb, S_LM)));

        // ── Split-1 `\p{N}{1,3}`: the one non-local rule — a group boundary every 3 chars from the
        // run start. Marker iteration (the paper's bounded-repetition lowering); ≤21 rounds/block,
        // and 0 rounds on text without digit runs.
        let mut m = n_start;
        if cy.dig_run && has(pb, S_N) {
            // resume mid-run: the next group start is (3 - since) chars into the block. Re-mask
            // with `num` at every hop — the carry only says the char *at* the edge was a digit, the
            // run may well have ended there (`Ⅷ` straddling the edge, then `\t`, then `456`).
            let mut s = lead & lead.wrapping_neg() & num; // first lead of the block
            for _ in 0..((3 - cy.dig_since % 3) % 3) {
                s = adv(s, b.cont) & num & lead;
            }
            m |= s;
        }
        let mut groups = m;
        if num & b.cont == 0 {
            // Fast path: every digit in this block is single-byte, so "3 chars on" is just `<< 3`
            // and the three `adv`s collapse into one shift against a precomputed mask (`num3` asks
            // that the two skipped positions are digits too, which is what the `adv` chain checked).
            // ~3 ops per group instead of ~14 — dense-digit text is otherwise this loop's worst case.
            let num3 = num & (num << 1) & (num << 2);
            while m != 0 {
                m = (m << 3) & num3;
                groups |= m;
            }
        } else {
            while m != 0 {
                let a = adv(m, b.cont) & num & lead;
                let c = adv(adv(a, b.cont) & num & lead, b.cont) & num & lead;
                if c == 0 {
                    break;
                }
                groups |= c;
                m = c;
            }
        }

        // ── whitespace: `\s*[\r\n]+ | \s+(?!\S) | \s+`.
        // (a) the run's first token runs through its LAST newline → a start right after it, unless
        //     a further newline still follows inside the run (a backward scan, hence the reversal).
        let anl = p1(s_nl, has(pb, S_NL)) & s_ws & lead;
        let later_nl = fill_to_last(s_nl.reverse_bits(), s_ws.reverse_bits()).reverse_bits();
        let after_nl = anl & !later_nl;
        // (b) the run's last char is handed to whatever follows, as its `[^…]?` / ` ?` prefix —
        //     unless the run ends the input or the next piece is Split-1/2-isolated (`(?!\S)`).
        let eof_bit = if last_blk { 1u64 << (len - 1) } else { 0 };
        let steal_lb = s_ws
            & !s_nl
            & lb
            & !eof_bit
            & !n1(s_ws, has(nb, S_WS))
            & !n1(s_n | cjk, has(nb, S_N | S_CJK));
        let (steal, steal_patch) = to_lead(steal_lb, b.cont, cy.cont);
        // (c) the same one-char give-back out of a gap run (Control / NumericOther / ZWJ match no
        //     alternative, so the run is one piece minus the char a following letter run claims).
        let (gap_steal, gap_patch) = to_lead(
            gap & lb & n1(lm, has(nb, S_LM) && !has(nb, S_CJK)),
            b.cont,
            cy.cont,
        );

        // ── alt-1 `[ascii_punct][A-Za-z]+`: fires only where the scan is actually positioned, i.e.
        // at a punct-run start no space swallowed. Its `[A-Za-z]+` run then has no interior starts
        // and forces one at its end — which the letter rule alone would suppress (`!c` reads `!ab`,
        // `c` after it is a fresh token even though its predecessor is a letter).
        let alt1 = ps_start & ascii & n1(aa, nb_aa);
        let aa_m = ((alt1 as u128) << 1) | u128::from(cy.aa_run);
        let aa_e = scanthru(aa_m, aa as u128);
        let aa_span = aa_e.wrapping_sub(aa_m);
        // ── a punct run's `[\r\n]*` tail swallows the newlines directly behind it.
        let nl_m =
            ((p1(ps, has(pb, S_PS) && !pb_cjk) & s_nl & lead) as u128) | u128::from(cy.nl_run);
        let nl_e = scanthru(nl_m, s_nl as u128);
        let nl_span = nl_e.wrapping_sub(nl_m);

        // ── the one backward-in-time dependency: a newline arriving now retracts the "start after
        // the last newline" already committed for a whitespace run that was open at the last edge.
        if let Some(p) = cy.anl
            && has(pb, S_WS)
            && s_ws & 1 != 0
            && s_nl & lead_run(s_ws, valid) != 0
        {
            starts[p / 64] &= !(1u64 << (p % 64));
            cy.anl = None;
        }

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
        st &= lead;
        if bi == 0 {
            st |= 1; // position 0 always opens a token
        }
        starts[bi] = st;
        if bi > 0 {
            starts[bi - 1] |= steal_patch | gap_patch;
        }

        // ── carries ────────────────────────────────────────────────────────────────────────────
        cy.aa_run = aa_e >> 64 != 0;
        cy.nl_run = nl_e >> 64 != 0;
        let tn = trail_run(num, valid, len);
        cy.dig_run = tn != 0 && has(nb, S_N) && !has(nb, S_CJK);
        cy.dig_since = if !cy.dig_run {
            0
        } else {
            let g = groups & tn;
            let counted = if g == 0 {
                cy.dig_since + (num & lead & tn).count_ones()
            } else {
                (num & lead & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
            };
            counted % 3
        };
        let tws = trail_run(s_ws, valid, len);
        if tws != 0 && has(nb, S_WS) {
            // ...but not a bit that a punct tail's `[\r\n]*` already made a *run start*: the
            // absorption cut the whitespace run, so a later newline cannot reach back past it.
            let a = after_nl & tws & !(nl_e as u64);
            if a != 0 {
                cy.anl = Some(base + 63 - a.leading_zeros() as usize);
            } else if !(tws & 1 != 0 && has(pb, S_WS)) {
                cy.anl = None; // a fresh run with no newline yet — nothing left to retract
            }
        } else {
            cy.anl = None;
        }
        cy.code = last_code;
        cy.cjk = last_cjk;
        cy.cont = b.cont;
    }

    emit(starts, nblk, ntext, out)
}
