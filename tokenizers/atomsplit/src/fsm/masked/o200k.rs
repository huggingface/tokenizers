//! Masked scheme for the o200k regex family (o200k / GPT-4o / gpt-oss with contractions and
//! digit cap 3, Mistral tekken without contractions at cap 1) — the boundary algebra via
//! gigatoken's `o200k_family.rs` (MIT) over tag-fed class masks.
//!
//! Differences from the cl100k family:
//! - Letter runs are case-structured. Under leftmost-greedy backtracking the two letter
//!   alternatives reduce to a phase automaton: a strict-upper char (`\p{Lu}\p{Lt}`) ends a
//!   token exactly when the previous char is strict-lower (`\p{Ll}`) — "camelCase" splits
//!   `camel|Case`, "HTTPResponse" stays one token. A strict-upper after a CASELESS letter
//!   needs the phase and lookahead (the backtrack to the last caseless char), so those chars
//!   defer to the scalar path.
//! - Contractions are attached suffixes of letter tokens, not a standalone alternative:
//!   "don't" is ONE token and the char after a consumed suffix always starts a new one
//!   ("can'ts" is `can't|s`). A contraction applies only when the apostrophe directly follows
//!   a letter-run char; elsewhere `'` is ordinary punctuation.
//! - Punct runs absorb a `[\r\n/]*` tail. `/` is itself punct, so an absorbed tail always
//!   begins with a newline; whether a batch-leading `[\r\n/]` run continues such a tail is
//!   resolved by a bounded walkback over the preceding text.
//! - Marks (`\p{M}`) are dual-class: they join letter runs AND continue punct runs, so their
//!   effective class is run-contextual. Mark chars (rare) defer to the scalar path with a bad
//!   smear wide enough (8 bytes forward) to cover every boundary their class can influence
//!   (two chars of multi-byte followers).

use super::super::o200k::advance_o200k;
use super::block::Block;
use super::{MaskedFsm, char_lead, cont_runs, digit_run_splits3, fill, shl_sat, smear_up};
use crate::fsm::{APO, ASM, Atom, CONT, LET, MRK, NLN, NO, NW, SPC, WSO, ZWJ, in_mask, mask};

pub(super) struct O200kMasked<const CONTRACTION: bool, const DIGIT_CAP: usize>;

impl<const CONTRACTION: bool, const DIGIT_CAP: usize> MaskedFsm
    for O200kMasked<CONTRACTION, DIGIT_CAP>
{
    #[inline(always)]
    fn batch_masks(&self, text: &[u8], tags: &[u8], scan: usize) -> (u64, u64) {
        batch_masks::<CONTRACTION, DIGIT_CAP>(text, tags, scan)
    }

    #[inline(always)]
    fn advance(&self, text: &[u8], tags: &[u8], i: usize, end: usize) -> usize {
        advance_o200k::<CONTRACTION, DIGIT_CAP>(text, tags, i, end)
    }
}

#[inline(always)]
fn is_tail_byte(b: u8) -> bool {
    matches!(b, b'\r' | b'\n' | b'/')
}

#[inline(always)]
fn is_member_mark(t: u8) -> bool {
    t & 0x0F == MRK && t != ASM && t != ZWJ
}

/// Was the tail-class byte at `scan - 1` absorbed by a punct run's `[\r\n/]*` tail (as opposed
/// to being a fresh punct-run `/` or a ws-run newline)? Walks the tail-class run back (bounded)
/// and classifies the char before it. `None`: unresolved (over-long run, or a preceding mark
/// whose own class is run-contextual).
fn prev_tail_absorbed(text: &[u8], tags: &[u8], scan: usize) -> Option<bool> {
    debug_assert!(scan >= 1 && is_tail_byte(text[scan - 1]));
    let mut r = scan - 1;
    let mut steps = 0;
    while r > 0 && is_tail_byte(text[r - 1]) {
        r -= 1;
        steps += 1;
        if steps > 8 {
            return None;
        }
    }
    // T-run = text[r..scan]. The `[\r\n/]*` tail is greedy, so once absorption triggers — at
    // the first newline that directly follows a punct-run char (an in-run slash, or the
    // pre-run char for a run-leading newline) — everything to the run's end is absorbed.
    // Before the trigger, newlines are ws-run members and slashes ordinary punct-run bytes.
    let run = &text[r..scan];
    let mut trigger = usize::MAX;
    let mut seen_slash = false;
    for (j, &b) in run.iter().enumerate() {
        if b == b'/' {
            seen_slash = true;
            continue;
        }
        if seen_slash {
            trigger = j;
            break;
        }
        if j == 0 {
            if r == 0 {
                continue;
            }
            let t = tags[char_lead(tags, r - 1)];
            if is_member_mark(t) || t & 0x0F >= 13 {
                // A mark continues whatever run precedes it; Sentinel/MultiByte are opaque.
                return None;
            }
            if !in_mask(t, mask::LETTER | mask::NUMBER | mask::WS) {
                trigger = 0;
                break;
            }
        }
    }
    Some(scan - 1 - r >= trigger)
}

/// Two-back "punct or space" test for the char whose lead is `p2`. A slash may be an absorbed
/// tail byte (a token end, neither punct-run member nor space), so it resolves through the
/// walkback. `None`: unresolved (the caller sets `force_bad_lead`). A mark P2 answers 0: the
/// bits that read it wrongly sit inside the previous batch's mark smear, and the scalar
/// overrun from there covers them (the walker's resume masking).
fn c2_os_at(text: &[u8], tags: &[u8], p2: usize) -> Option<u64> {
    if text[p2] == b'/' {
        return prev_tail_absorbed(text, tags, p2 + 1).map(|abs| u64::from(!abs));
    }
    let t2 = tags[p2];
    Some(u64::from(
        t2 & 0x0F == SPC || (!is_member_mark(t2) && in_mask(t2, mask::NOT_WS_L_N)),
    ))
}

/// Boundary carries from the two chars before the batch (the cl100k set, plus the case classes
/// and the absorbed-tail resolution).
#[derive(Default)]
struct Carries {
    pl: u64,
    pu: u64,
    pcl: u64,
    ps: u64,
    pwt: u64,
    po: u64,
    pws: u64,
    pd: u64,
    /// P1 is a member-mark (seeds the mark smear at bit 0).
    pmk: u64,
    c2_os: u64,
    b2b_in: u64,
    /// P1 is an absorbed `[\r\n/]*` tail byte whose token may continue into this batch.
    p_abs: bool,
    /// The tail walkback could not resolve: the batch's leading tail-class run (plus the byte
    /// after it) can't be trusted.
    force_bad_lead: bool,
}

fn carries(text: &[u8], tags: &[u8], scan: usize, c: u64) -> Carries {
    let mut cr = Carries::default();
    if scan == 0 {
        return cr;
    }
    if is_tail_byte(text[scan - 1]) {
        // An absorbed tail ended the previous token, so every "P1 is X" carry is zero and only
        // the tail-continuation seed survives. A fresh `/` is an ordinary punct byte; fresh
        // `\r\n` are ws-run newlines.
        match prev_tail_absorbed(text, tags, scan) {
            None => cr.force_bad_lead = true,
            Some(true) => cr.p_abs = true,
            Some(false) => {
                if text[scan - 1] == b'/' {
                    cr.po = 1;
                } else {
                    cr.pws = 1;
                }
                match c2_os_at(text, tags, char_lead(tags, scan - 2)) {
                    Some(v) => cr.c2_os = v,
                    None => cr.force_bad_lead = true,
                }
            }
        }
        return cr;
    }
    let p1 = char_lead(tags, scan - 1);
    let c2v = if p1 == 0 {
        Some(0)
    } else {
        c2_os_at(text, tags, char_lead(tags, p1 - 1))
    };
    match c2v {
        Some(v) => {
            if c & 1 != 0 {
                cr.b2b_in = v << (!c).trailing_zeros();
            } else {
                cr.c2_os = v;
            }
        }
        None => cr.force_bad_lead = true,
    }
    let t1 = tags[p1];
    match t1 & 0x0F {
        LET => {
            cr.pl = 1;
            cr.pu = u64::from(t1 == Atom::UpperLetter as u8);
            cr.pcl = u64::from(t1 == Atom::Letter as u8);
        }
        NW | NO => cr.pd = 1,
        SPC => {
            cr.pws = 1;
            cr.ps = 1;
        }
        WSO => {
            cr.pws = 1;
            cr.pwt = 1;
        }
        MRK if is_member_mark(t1) => cr.pmk = 1,
        _ => cr.po = 1,
    }
    cr
}

fn batch_masks<const CONTRACTION: bool, const DIGIT_CAP: usize>(
    text: &[u8],
    tags: &[u8],
    scan: usize,
) -> (u64, u64) {
    debug_assert!(scan + 64 < tags.len() && tags.len() == text.len());
    // SAFETY: `scan + 64 < tags.len()` (walker guarantee), the block's load contract.
    let blk = unsafe { Block::load(tags, scan) };
    if blk.any_range_tag(13, 1) {
        return (0, u64::MAX);
    }
    let l0 = blk.eq_tag(LET);
    let ub0 = blk.eq_full(Atom::UpperLetter as u8);
    let cl0 = blk.eq_full(Atom::Letter as u8);
    let mk0 = blk.eq_full(MRK); // true marks: refined Mark tags (ASM/ZWJ) are punct-class
    let d0 = blk.range_tag(NW, 1);
    let nl = blk.eq_tag(NLN);
    let s = blk.eq_tag(SPC);
    let wt0 = blk.eq_tag(WSO);
    let ap = if blk.any_eq_full(APO) {
        blk.eq_full(APO)
    } else {
        0
    };
    let c = if blk.any_eq_full(CONT) {
        blk.eq_full(CONT)
    } else {
        0
    };

    let cr = carries(text, tags, scan, c);

    let (mut l, mut u, mut clb, mut d, mut wt, mut mk) = (l0, ub0, cl0, d0, wt0, mk0);
    let (c2, c3) = cont_runs(c);
    if c != 0 {
        if c & 1 != 0 {
            let lead_in = c & ((1u64 << (!c).trailing_zeros()) - 1);
            l |= lead_in * cr.pl;
            u |= lead_in * cr.pu;
            clb |= lead_in * cr.pcl;
            d |= lead_in * cr.pd;
            wt |= lead_in * cr.pwt;
            mk |= lead_in * cr.pmk;
        }
        l = fill(l, c, c2, c3);
        u = fill(u, c, c2, c3);
        clb = fill(clb, c, c2, c3);
        d = fill(d, c, c2, c3);
        wt = fill(wt, c, c2, c3);
        mk = fill(mk, c, c2, c3);
    }
    mk |= cr.pmk; // a mark P1 poisons bit 0 even when it ends exactly at the batch edge
    let lead = !c;
    let len1 = lead & !(c >> 1);
    let len2 = lead & (c >> 1) & !(c >> 2);
    let len3 = lead & (c >> 1) & (c >> 2) & !(c >> 3);
    let len4 = lead & (c >> 1) & (c >> 2) & (c >> 3);
    let w2 = wt0 & len2;
    let w3 = wt0 & len3;

    let ws_f = s | nl | wt;
    // Marks land in the complement (dual-class); every bit that can read them is inside the
    // mark bad smear below, so their punct-class reading is never trusted.
    let o = !(l | d | ws_f);

    // --- Absorbed `[\r\n/]*` tails ---------------------------------------------------------
    // The tail class needs the slash mask from the TEXT block; skip that load when the batch
    // has no newline and no tail context carried in.
    let (tcls, abs_t) = if nl != 0 || cr.p_abs || cr.force_bad_lead {
        // SAFETY: `scan + 64 < text.len()` (walker guarantee), the block's load contract.
        let sl = unsafe { Block::load(text, scan) }.eq_full(b'/');
        let tcls = nl | sl;
        let abs_seed = (nl & ((o << 1) | cr.po)) | (u64::from(cr.p_abs) & tcls);
        let abs_t = if abs_seed == 0 {
            0
        } else {
            smear_up(abs_seed, tcls)
        };
        (tcls, abs_t)
    } else {
        (nl, 0)
    };
    let ob_eff = o & !abs_t;

    // --- Letters (see the cl100k scheme for the base rules) ---------------------------------
    let c_test = ((ob_eff | s) << 1) | cr.po | cr.ps;
    let b2back = ((c_test & len1) << 1)
        | ((c_test & len2) << 2)
        | ((c_test & len3) << 3)
        | ((c_test & len4) << 4)
        | cr.c2_os
        | cr.b2b_in;
    let p_l = (l << 1) | cr.pl;
    let p_u = (u << 1) | cr.pu;
    let p_cl = (clb << 1) | cr.pcl;
    let p_s = (s << 1) | cr.ps;
    let p_wt = (wt << 1) | cr.pwt;
    let p_o = (ob_eff << 1) | cr.po;
    let absorb = p_o & !b2back;
    // Casing boundary: a strict-upper char after a strict-lower one. (For ASCII text this is
    // the whole rule; upper-after-caseless defers below.)
    let p_sl = p_l & !p_u & !p_cl;
    let b_letters = (l0 & !p_l & !p_s & !p_wt & !absorb) | (ub0 & p_sl);

    // --- Digits ------------------------------------------------------------------------------
    let d_ascii = d0 & len1;
    let dmb = (d & c) | (d0 & !len1);
    let b_digits = if DIGIT_CAP == 3 {
        if d_ascii & (d_ascii >> 1) != 0 {
            digit_run_splits3(d_ascii)
        } else {
            d_ascii
        }
    } else {
        d0 // cap 1: every digit char its own token
    };

    // --- Punct: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` --------------------------------------------------
    let b_punct = (ob_eff & lead) & !p_o & !p_s;

    // --- Bad zones ----------------------------------------------------------------------------
    let mut bad = if DIGIT_CAP == 3 {
        dmb | dmb << 1 | dmb >> 1
    } else {
        0
    };
    if mk != 0 {
        // A mark's run-contextual class can affect boundaries up to two chars after it (8
        // bytes of multi-byte followers) and the byte before.
        bad |= mk
            | (mk << 1)
            | (mk << 2)
            | (mk << 3)
            | (mk << 4)
            | (mk << 5)
            | (mk << 6)
            | (mk << 7)
            | (mk << 8)
            | (mk >> 1);
    }
    // A strict-upper char after a caseless letter: phase- and lookahead-dependent.
    bad |= ub0 & ((clb << 1) | cr.pcl);
    if cr.force_bad_lead {
        bad |= (smear_up(tcls & 1, tcls) << 1) | 0b11;
    }

    // --- Whitespace ---------------------------------------------------------------------------
    let ws_eff = ws_f & !abs_t;
    let la = tags[scan + 64] & 0x0F;
    let nn64 = la != CONT && !in_mask(la, mask::WS);
    let nn64m = u64::from(nn64).wrapping_neg();

    // An absorbed tail touching the batch end continues iff byte 64 is tail-class; the next
    // batch's tail walkback re-derives the context either way, so nothing defers here. A ws
    // run touching the batch end still defers when byte 64 is ws.
    let nonws = !ws_eff;
    if ws_eff >> 63 != 0 && !nn64 {
        if nonws == 0 {
            return (0, u64::MAX);
        }
        let h = 63 - nonws.leading_zeros();
        bad |= u64::MAX << (h + 1);
    }
    if DIGIT_CAP == 3 {
        let seed = (d_ascii & (bad << 1)) | (d_ascii & cr.pd);
        if seed != 0 {
            bad |= smear_up(seed, d_ascii);
        }
    }

    let ws_leads1 = (s | nl | (wt0 & len1)) & ws_eff;
    let ws_leads = (ws_leads1 | w2 | w3) & !abs_t;
    let p_ws = (ws_eff << 1) | cr.pws;
    let edge_last = (ws_leads1 & (1 << 63)) | (w2 & (1 << 62)) | (w3 & (1 << 61));
    let split_ok = (ws_leads1 & (nonws >> 1))
        | (w2 & (nonws >> 2))
        | (w3 & (nonws >> 3))
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
        if q + 1 < e {
            b_ws |= 1u64 << (q + 1);
            let tail = run_mask & (u64::MAX << (q + 1));
            let tail_leads = ws_leads & tail;
            b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
        }
        runs_n &= !run_mask;
    }

    let mut boundary = b_letters | b_digits | b_punct | b_ws;

    // --- Contractions: suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` ---------------------------------
    // An apostrophe at a boundary right after a letter-run char merges the suffix into that
    // token and forces a boundary right after it.
    if CONTRACTION {
        let mut cand = ap & boundary & p_l & !bad;
        let mut last_forced = usize::MAX;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i <= 2 {
                // The preceding letter could itself end an earlier contraction that started
                // before the batch: scalar.
                bad |= 0b111u64 << i;
                continue;
            }
            if i >= 61 {
                bad |= u64::MAX << i;
                break;
            }
            if i == last_forced {
                // "x'll'd": the letter before this apostrophe is a consumed suffix's last
                // char; a new (prefix) match starts here instead.
                continue;
            }
            // The letter before this apostrophe may itself be a consumed suffix's last char
            // resolved where `last_forced` can't see it (a scalar-walked zone, or a fixup
            // before the batch): locally ambiguous, defer.
            let p = scan + i;
            let prev_suffix_possible = (text[p - 2] == b'\''
                && matches!(text[p - 1] | 0x20, b's' | b'd' | b'm' | b't'))
                || (text[p - 3] == b'\''
                    && (matches!(
                        (text[p - 2] | 0x20, text[p - 1] | 0x20),
                        (b'l', b'l') | (b'v', b'e') | (b'r', b'e')
                    ) || (text[p - 2] == 0xC5 && text[p - 1] == 0xBF)));
            if prev_suffix_possible {
                bad |= 0b111u64 << i;
                continue;
            }
            let b1 = text[p + 1];
            if b1 >= 0x80 {
                // `(?i:'s)` also matches 'ſ (U+017F): defer.
                bad |= 0b111u64 << i;
                continue;
            }
            let k = match b1 | 0x20 {
                b's' | b'd' | b'm' | b't' => 2,
                b'l' if text[p + 2] | 0x20 == b'l' => 3,
                b'v' if text[p + 2] | 0x20 == b'e' => 3,
                b'r' if text[p + 2] | 0x20 == b'e' => 3,
                _ => 0,
            };
            if k != 0 {
                boundary &= !(1u64 << i);
                boundary &= !(((1u64 << (k - 1)) - 1) << (i + 1));
                boundary |= 1u64 << (i + k);
                last_forced = i + k;
            }
        }
    }

    (boundary & !bad, bad)
}
