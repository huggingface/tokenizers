//! Masked scheme for the deepseek-v3 Sequence (digits `\p{N}{1,3}` → CJK-range runs → the big
//! regex) — boundary rules derived from [`scan_deepseek`]'s dispatch, following gigatoken's
//! `deepseek_v3.rs` scoping (MIT).
//!
//! Deepseek has no case rules and no contractions, but three shapes of its own:
//! - The alt-2 prefix class is `[^\r\n\p{L}\p{P}\p{S}]?`: letters absorb a preceding space,
//!   non-newline whitespace char, or the LAST char of a gap run (Control / NumericOther /
//!   ZWJ — chars matching no alternative), and never a punct char or a digit.
//! - alt-1 `[ascii-punct][A-Za-z]+`: an ASCII punct char at a token start absorbs a following
//!   ASCII-letter run. Where that run collides with a non-ASCII letter or mark, the two rules
//!   diverge (`[A-Za-z]+` stops, `[\p{L}\p{M}]+` would continue); those collision bits defer.
//! - A whitespace run followed by a digit or CJK char is one whole token (Split-1/2 isolated
//!   the follower), so the `\s+(?!\S)` give-back is gated on the follower's class.
//!
//! CJK-range chars are closed units re-split into same-kind sub-runs, and every other rule
//! stops at them; a batch containing any byte with a lead in `0xE3..=0xE9` (a superset of the
//! CJK ranges) defers whole, before the tag masks are built. On CJK-dominated text the scanner
//! is then the scalar scan plus one 64-byte test per batch, which is the accepted trade: the
//! win is on latin/code text, and CJK batches resolve through the same scalar rules as before.

use super::super::deepseek::advance_deepseek;
use super::block::Block;
use super::{MaskedFsm, char_lead, cont_runs, digit_run_splits3, fill, shl_sat, smear_up};
use crate::fsm::{ASM, CON, CONT, CTL, LET, MRK, NLN, NMO, NO, NW, SPC, WSO, ZWJ, in_mask, mask};

pub(super) struct DeepSeekMasked;

impl MaskedFsm for DeepSeekMasked {
    #[inline(always)]
    fn batch_masks(&self, text: &[u8], tags: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(text, tags, scan)
    }

    #[inline(always)]
    fn advance(&self, text: &[u8], tags: &[u8], i: usize, end: usize) -> usize {
        advance_deepseek(text, tags, i, end)
    }
}

/// A deepseek letter-run member tag: `[\p{L}\p{M}]` minus the refined Marks (ASM/ZWJ). The
/// CJK-range exclusion is handled by the bad cover, not here.
#[inline(always)]
fn is_member(t: u8) -> bool {
    in_mask(t, mask::LETTER_MARK) && t != ASM && t != ZWJ
}

#[derive(Default)]
struct Carries {
    pl: u64,
    ps: u64,
    pwt: u64,
    po: u64,
    pws: u64,
    pd: u64,
    pgap: u64,
    /// Byte `scan - 1` is an ASCII punct char at a token start: an alt-1 absorb may reach into
    /// this batch.
    alt1: u64,
    /// P1 is a CJK-range char. Its all-zero carries say "closed unit", which is right for the
    /// char AFTER it; bytes of the char itself straddling into the batch stay unclassified and
    /// must defer (see `batch_masks`).
    cjk: bool,
}

fn carries(text: &[u8], tags: &[u8], scan: usize) -> Carries {
    let mut cr = Carries::default();
    if scan == 0 {
        return cr;
    }
    let p1 = char_lead(tags, scan - 1);
    if (0xE3..=0xE9).contains(&text[p1]) {
        // A CJK-range char is a closed unit: everything after it starts fresh, which is what
        // all-zero carries say.
        cr.cjk = true;
        return cr;
    }
    let t1 = tags[p1];
    match t1 & 0x0F {
        LET | MRK if is_member(t1) => cr.pl = 1,
        NW | NO => cr.pd = 1,
        SPC => {
            cr.pws = 1;
            cr.ps = 1;
        }
        WSO => {
            cr.pws = 1;
            cr.pwt = 1;
        }
        NLN => cr.pws = 1,
        NMO | CTL => cr.pgap = 1,
        _ if t1 == ZWJ => cr.pgap = 1,
        _ => cr.po = 1,
    }
    if text[scan - 1].is_ascii_punctuation() {
        // Was P1 (one byte) at a token start? Not when it continues a punct run or follows a
        // space; a CJK-range P2 is a closed unit, so P1 starts fresh after it.
        let at_start = if p1 == 0 {
            true
        } else {
            let p2 = char_lead(tags, p1 - 1);
            let t2 = tags[p2];
            (0xE3..=0xE9).contains(&text[p2])
                || !(t2 & 0x0F == SPC || in_mask(t2, mask::PUNCT_SYM) || t2 == ASM)
        };
        cr.alt1 = u64::from(at_start);
    }
    cr
}

fn batch_masks(text: &[u8], tags: &[u8], scan: usize) -> (u64, u64) {
    debug_assert!(scan + 64 < tags.len() && tags.len() == text.len());
    // SAFETY: `scan + 64 < tags.len()` (walker guarantee), the blocks' load contract.
    let tb = unsafe { Block::load(text, scan) };
    // CJK-range chars (lead 0xE3..=0xE9, a superset of the Split-2 ranges) are closed units
    // that every other rule stops at; a batch containing any defers whole, BEFORE the tag
    // masks are built. On CJK-dominated text the scanner degenerates to the scalar scan plus
    // this one test (see the module doc).
    if tb.range_full(0xE3, 6) != 0 {
        return (0, u64::MAX);
    }
    // SAFETY: `scan + 64 < tags.len()` (walker guarantee), the block's load contract.
    let blk = unsafe { Block::load(tags, scan) };
    if blk.any_range_tag(13, 1) {
        return (0, u64::MAX);
    }
    let l0 = blk.eq_tag(LET);
    let mk0 = blk.eq_full(MRK); // true marks join deepseek letter runs
    let o0 = blk.range_tag(CON, 3) | blk.eq_full(ASM); // `[\p{P}\p{S}]` = Con|Pun|Apo|Sym (+ASM)
    let gap0 = blk.range_tag(NMO, 1) | blk.eq_full(ZWJ);
    let d0 = blk.range_tag(NW, 1);
    let nl = blk.eq_tag(NLN);
    let s = blk.eq_tag(SPC);
    let wt0 = blk.eq_tag(WSO);
    let c = if blk.any_eq_full(CONT) {
        blk.eq_full(CONT)
    } else {
        0
    };
    let pa = tb.ascii_punct();
    let alpha = tb.ascii_alpha();

    let cr = carries(text, tags, scan);

    let lml0 = l0 | mk0;
    let (mut lm, mut d, mut wt, mut g) = (lml0, d0, wt0, gap0);
    let (c2, c3) = cont_runs(c);
    if c != 0 {
        if c & 1 != 0 {
            let lead_in = c & ((1u64 << (!c).trailing_zeros()) - 1);
            lm |= lead_in * cr.pl;
            d |= lead_in * cr.pd;
            wt |= lead_in * cr.pwt;
            g |= lead_in * cr.pgap;
        }
        lm = fill(lm, c, c2, c3);
        d = fill(d, c, c2, c3);
        wt = fill(wt, c, c2, c3);
        g = fill(g, c, c2, c3);
    }
    let lead = !c;
    let len1 = lead & !(c >> 1);
    let len2 = lead & (c >> 1) & !(c >> 2);
    let len3 = lead & (c >> 1) & (c >> 2) & !(c >> 3);
    let len4 = lead & (c >> 1) & (c >> 2) & (c >> 3);
    let w2 = wt0 & len2;
    let w3 = wt0 & len3;

    let ws_f = s | nl | wt;
    // The complement is the punct-run class here too (gap and CJK bytes land in it; both are
    // covered by their own masks or the bad cover below).
    let o = !(lm | d | ws_f | g);

    let mut bad = 0u64;
    if cr.cjk && c & 1 != 0 {
        // A CJK char straddling into the batch: its continuation bytes carry no class (the
        // all-zero carries only cover the char after it), so they and the char right after
        // them defer.
        let e1 = (!c).trailing_zeros();
        bad |= (c & ((1u64 << e1) - 1)) | (1u64 << e1);
    }

    // --- Letters: `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` and alt-1 `[ascii-punct][A-Za-z]+` ---
    let p_lm = (lm << 1) | cr.pl;
    let p_s = (s << 1) | cr.ps;
    let p_wt = (wt << 1) | cr.pwt;
    let p_g = (g << 1) | cr.pgap;
    let p_o = (o << 1) | cr.po;

    // --- Punct: ` ?[\p{P}\p{S}]+[\r\n]*` -------------------------------------------------------
    let b_punct = o0 & !p_o & !p_s;
    let abs_seed = nl & ((o << 1) | cr.po);
    let abs_n = if abs_seed == 0 {
        0
    } else {
        smear_up(abs_seed, nl)
    };
    let ws_eff = ws_f & !abs_n;

    // alt-1 absorbs: an ASCII-letter byte right after a token-starting ASCII punct char, and
    // the rest of that `[A-Za-z]+` run. Where the run's end meets a letter-run member the two
    // letter rules diverge: defer that bit. A run touching the batch end defers too (the next
    // batch cannot tell an alt-1 run from a plain letter run).
    let absorb_a = alpha & (((pa & b_punct) << 1) | cr.alt1);
    let b_letters = lml0 & !p_lm & !p_s & !p_wt & !p_g & !absorb_a;
    if absorb_a != 0 {
        let zone = smear_up(absorb_a, alpha);
        bad |= (zone << 1) & !alpha & lml0;
        if zone >> 63 != 0 {
            bad |= 1u64 << 63;
        }
    }

    // --- Gap runs: boundary at the run start, and at the last char before an absorbed letter
    // run (the prefix split; one and the same bit for a single-char gap). The prefix split
    // reads the NEXT char's lead, so a gap char whose follower sits past the batch edge
    // defers.
    let b_gap = (gap0 & !p_g)
        | (gap0
            & ((len1 & (lml0 >> 1))
                | (len2 & (lml0 >> 2))
                | (len3 & (lml0 >> 3))
                | (len4 & (lml0 >> 4))));
    bad |= (gap0 & len1 & (u64::MAX << 63))
        | (gap0 & len2 & (u64::MAX << 62))
        | (gap0 & len3 & (u64::MAX << 61))
        | (gap0 & len4 & (u64::MAX << 60));

    // --- Digits: `\p{N}{1,3}` (the cl100k cap-3 machinery) -------------------------------------
    let d_ascii = d0 & len1;
    let dmb = (d & c) | (d0 & !len1);
    let b_digits = if d_ascii & (d_ascii >> 1) != 0 {
        digit_run_splits3(d_ascii)
    } else {
        d_ascii
    };
    bad |= dmb | dmb << 1 | dmb >> 1;

    // --- Whitespace -----------------------------------------------------------------------------
    // The give-back is gated on the follower: a digit or CJK char after the run means the whole
    // run is one token (`iso` covers the follower's bytes; CJK is inside `bad` anyway, but the
    // gate keeps the algebra honest about why).
    let la = tags[scan + 64] & 0x0F;
    let la_iso = matches!(la, NW | NO) || (0xE3..=0xE9).contains(&text[scan + 64]);
    let nn64 = la != CONT && !in_mask(la, mask::WS);
    let nn64m = u64::from(nn64 && !la_iso).wrapping_neg();
    if abs_n >> 63 != 0 && !nn64 {
        bad |= 1u64 << 63;
    }
    let nonws = !ws_eff;
    if ws_eff >> 63 != 0 && !nn64 {
        if nonws == 0 {
            return (0, u64::MAX);
        }
        let h = 63 - nonws.leading_zeros();
        bad |= u64::MAX << (h + 1);
    }
    let seed = (d_ascii & (bad << 1)) | (d_ascii & cr.pd);
    if seed != 0 {
        bad |= smear_up(seed, d_ascii);
    }

    // A whitespace run followed by a digit keeps its last char (no give-back); the CJK case
    // never reaches this path (any in-batch CJK deferred above), leaving only the lookahead.
    let iso = d;
    let nonws_ni = nonws & !iso;
    let p_ws = (ws_eff << 1) | cr.pws;
    let ws_leads1 = (s | nl | (wt0 & len1)) & ws_eff;
    let ws_leads = (ws_leads1 | w2 | w3) & !abs_n;
    let edge_last = (ws_leads1 & (1 << 63)) | (w2 & (1 << 62)) | (w3 & (1 << 61));
    let split_ok = (ws_leads1 & (nonws_ni >> 1))
        | (w2 & (nonws_ni >> 2))
        | (w3 & (nonws_ni >> 3))
        | (edge_last & nn64m);
    let mut b_ws = ws_leads & (!p_ws | split_ok);

    let mut runs_n = nl & ws_eff & !bad;
    while runs_n != 0 {
        let f = runs_n.trailing_zeros();
        let below_gap = nonws & ((1u64 << f) - 1);
        let a = if below_gap == 0 {
            0
        } else {
            64 - below_gap.leading_zeros()
        };
        let e = (nonws & (u64::MAX << f)).trailing_zeros();
        let run_mask = (u64::MAX << a) & !shl_sat(u64::MAX, e);
        b_ws &= !run_mask;
        b_ws |= 1u64 << a;
        let q = 63 - (nl & run_mask).leading_zeros();
        // The post-newline tail: no give-back when the run's follower is a digit or CJK char
        // (the whole tail is then one token).
        let follower_iso = if e >= 64 { la_iso } else { iso >> e & 1 != 0 };
        if q + 1 < e {
            b_ws |= 1u64 << (q + 1);
            if !follower_iso {
                let tail = run_mask & (u64::MAX << (q + 1));
                let tail_leads = ws_leads & tail;
                b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
            }
        }
        runs_n &= !run_mask;
    }

    let boundary = b_letters | b_digits | b_punct | b_gap | b_ws;
    (boundary & !bad, bad)
}
