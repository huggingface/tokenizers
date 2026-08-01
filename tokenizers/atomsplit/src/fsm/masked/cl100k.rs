//! Masked scheme for the cl100k regex family (cl100k / Llama-3 / GLM at digit cap 3, Qwen2 at
//! cap 1, unbounded `\p{N}+` at `usize::MAX`) — the boundary algebra via gigatoken's
//! `cl100k_family.rs` (MIT) over tag-fed class masks.
//!
//! Boundary rules (the regex is `'(?i:contractions)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,cap}|
//! ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`):
//! - A letter starts a token unless it continues a letter run, follows a space or a non-newline
//!   whitespace char (those always sit at a boundary before a non-ws char and absorb one letter
//!   run via the `[^\r\n\p{L}\p{N}]?` prefix), or follows a punct char that is itself at a
//!   boundary, i.e. whose own predecessor is neither punct nor space (a two-chars-back test,
//!   made char-aware by shifting per the previous char's byte length).
//! - Digits split every `cap` chars from each run start and never absorb a preceding space.
//! - A punct char starts a token unless it continues a punct run or follows a space.
//! - Newlines directly after a punct run are absorbed (`[\r\n]*`).
//! - A whitespace run containing newlines emits one token through its LAST newline
//!   (`\s*[\r\n]`), then the give-back rules; NL-free runs split before their last char when
//!   followed by non-ws (`\s+(?!\S)`).
//!
//! Bad zones: multi-byte `\p{N}` chars under cap 3 (the grouping is char-counted, byte hops
//! would misphase it) and any digit run whose phase starts before the batch; whitespace runs
//! touching the batch end while the next char is whitespace (their last newline may lie
//! beyond); apostrophes near the batch edge or before a non-ASCII char (`(?i:'s)` also matches
//! `'ſ`); `Sentinel`/`MultiByte` tags.

use super::super::cl100k::advance_cl100k_cap;
use super::block::Block;
use super::{MaskedFsm, char_lead, cont_runs, digit_run_splits3, fill, shl_sat, smear_up};
use crate::fsm::{APO, CONT, LET, NLN, NO, NW, SPC, WSO, in_mask, mask};

pub(super) struct Cl100kMasked {
    /// 1, 3 or `usize::MAX`; the entry point routes any other cap to the scalar scan.
    pub(super) digit_cap: usize,
}

impl MaskedFsm for Cl100kMasked {
    #[inline(always)]
    fn batch_masks(&self, text: &[u8], tags: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(text, tags, scan, self.digit_cap)
    }

    #[inline(always)]
    fn advance(&self, text: &[u8], tags: &[u8], i: usize, end: usize) -> usize {
        advance_cl100k_cap(text, tags, i, end, self.digit_cap)
    }
}

/// Boundary carries from the two chars before the batch: P1 is the char containing byte
/// `scan - 1`, P2 the one before it (the two-chars-back absorb test).
#[derive(Default)]
struct Carries {
    pl: u64,
    ps: u64,
    pwt: u64,
    po: u64,
    pws: u64,
    pd: u64,
    /// P2 is punct-or-space, for a char lead at bit 0 (P1 entirely before the batch).
    c2_os: u64,
    /// The same test positioned at the first lead after a P1 that straddles into the batch
    /// (P1's own predecessor is then P2).
    b2b_in: u64,
}

fn carries(tags: &[u8], scan: usize, c: u64) -> Carries {
    let mut cr = Carries::default();
    if scan == 0 {
        return cr;
    }
    let p1 = char_lead(tags, scan - 1);
    let c2v = if p1 == 0 {
        0
    } else {
        let t2 = tags[char_lead(tags, p1 - 1)] & 0x0F;
        u64::from(t2 == SPC || in_mask(t2, mask::NOT_WS_L_N))
    };
    if c & 1 != 0 {
        cr.b2b_in = c2v << (!c).trailing_zeros();
    } else {
        cr.c2_os = c2v;
    }
    match tags[p1] & 0x0F {
        LET => cr.pl = 1,
        NW | NO => cr.pd = 1,
        SPC => {
            cr.pws = 1;
            cr.ps = 1;
        }
        WSO => {
            cr.pws = 1;
            cr.pwt = 1;
        }
        // A newline is whitespace but never a `[^\r\n\p{L}\p{N}]?` prefix, so pwt stays 0.
        NLN => cr.pws = 1,
        _ => cr.po = 1,
    }
    cr
}

fn batch_masks(text: &[u8], tags: &[u8], scan: usize, digit_cap: usize) -> (u64, u64) {
    debug_assert!(scan + 64 < tags.len() && tags.len() == text.len());
    // SAFETY: `scan + 64 < tags.len()` (walker guarantee), the block's load contract.
    let blk = unsafe { Block::load(tags, scan) };
    if blk.any_range_tag(13, 1) {
        // Sentinel / MultiByte: the scalar dispatch has a defensive arm for these.
        return (0, u64::MAX);
    }
    let l0 = blk.eq_tag(LET);
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

    let cr = carries(tags, scan, c);

    // Fill: continuation bytes join their char's class; a char straddling into the batch has
    // its leading continuation bytes take P1's class (punct needs no action: it is derived as
    // the complement).
    let (mut l, mut d, mut wt) = (l0, d0, wt0);
    let (c2, c3) = cont_runs(c);
    if c != 0 {
        if c & 1 != 0 {
            let lead_in = c & ((1u64 << (!c).trailing_zeros()) - 1);
            l |= lead_in * cr.pl;
            d |= lead_in * cr.pd;
            wt |= lead_in * cr.pwt;
        }
        l = fill(l, c, c2, c3);
        d = fill(d, c, c2, c3);
        wt = fill(wt, c, c2, c3);
    }
    // Per-length char leads (`\s` chars are at most 3 bytes; letters up to 4).
    let lead = !c;
    let len1 = lead & !(c >> 1);
    let len2 = lead & (c >> 1) & !(c >> 2);
    let len3 = lead & (c >> 1) & (c >> 2) & !(c >> 3);
    let len4 = lead & (c >> 1) & (c >> 2) & (c >> 3);
    let w2 = wt0 & len2;
    let w3 = wt0 & len3;

    let ws_f = s | nl | wt;
    let o = !(l | d | ws_f);

    // --- Letters: `[^\r\n\p{L}\p{N}]?\p{L}+` --------------------------------------------------
    // b2back: "the char two back is punct or space", evaluated at each char's lead by shifting
    // the prev-byte test by the PREVIOUS char's byte length.
    let c_test = ((o | s) << 1) | cr.po | cr.ps;
    let b2back = ((c_test & len1) << 1)
        | ((c_test & len2) << 2)
        | ((c_test & len3) << 3)
        | ((c_test & len4) << 4)
        | cr.c2_os
        | cr.b2b_in;
    let p_l = (l << 1) | cr.pl;
    let p_s = (s << 1) | cr.ps;
    let p_wt = (wt << 1) | cr.pwt;
    let p_o = (o << 1) | cr.po;
    let absorb = p_o & !b2back;
    let b_letters = l0 & !p_l & !p_s & !p_wt & !absorb;

    // --- Digits: `\p{N}{1,cap}` ----------------------------------------------------------------
    // Only 1-byte digit chars can be split by byte hops; multi-byte `\p{N}` runs go to the
    // scalar path under cap 3 (bad below). Cap 1 tokens are single chars (any width), and the
    // unbounded cap only needs run starts.
    let d_ascii = d0 & len1;
    let dmb = (d & c) | (d0 & !len1);
    let b_digits = match digit_cap {
        3 => {
            if d_ascii & (d_ascii >> 1) != 0 {
                digit_run_splits3(d_ascii)
            } else {
                d_ascii
            }
        }
        1 => d0,
        _ => d0 & !((d << 1) | cr.pd),
    };

    // --- Punct: ` ?[^\s\p{L}\p{N}]+[\r\n]*` ----------------------------------------------------
    let b_punct = (o & lead) & !p_o & !p_s;

    // Newlines directly after a punct run are absorbed (`[\r\n]*`).
    let abs_seed = nl & ((o << 1) | cr.po);
    let abs_n = if abs_seed == 0 {
        0
    } else {
        smear_up(abs_seed, nl)
    };
    let ws_eff = ws_f & !abs_n;

    let mut bad = if digit_cap == 3 {
        dmb | dmb << 1 | dmb >> 1
    } else {
        0
    };

    // Byte-64 lookahead: is the char at the next batch's first byte non-ws? Decides whether
    // ws-like runs touching bit 63 resolve in-batch. A CONT lookahead means a char straddles
    // out; treating that as "ws" is safe: a live give-back at bit 63 would need the next char's
    // lead at byte 64, which contradicts the straddle.
    let la = tags[scan + 64] & 0x0F;
    let nn64 = la != CONT && !in_mask(la, mask::WS);
    let nn64m = u64::from(nn64).wrapping_neg();

    // An absorbed newline touching the batch end: if byte 64 is ws the token may continue with
    // another newline, and the next batch cannot tell an absorbed `\n` before its bit 0 from a
    // ws-run `\n` — defer.
    if abs_n >> 63 != 0 && !nn64 {
        bad |= 1u64 << 63;
    }

    // A ws run touching the batch end resolves in-batch only when byte 64's char is non-ws
    // (its last newline and `(?!\S)` split are then all visible); otherwise defer it.
    let nonws = !ws_eff;
    if ws_eff >> 63 != 0 && !nn64 {
        if nonws == 0 {
            return (0, u64::MAX); // whole batch one ws run
        }
        let h = 63 - nonws.leading_zeros();
        bad |= u64::MAX << (h + 1);
    }

    // A digit run whose grouping phase did not start inside this batch (a continuation from
    // before it, or following a bad zone that may hold digit chars) defers too.
    if digit_cap == 3 {
        let seed = (d_ascii & (bad << 1)) | (d_ascii & cr.pd);
        if seed != 0 {
            bad |= smear_up(seed, d_ascii);
        }
    }

    // --- Whitespace ---------------------------------------------------------------------------
    // Base rule (correct for NL-free runs; NL runs are overridden below): run start, or split
    // before the last char when followed by non-ws.
    let ws_leads1 = (s | nl | (wt0 & len1)) & ws_eff;
    let ws_leads = (ws_leads1 | w2 | w3) & !abs_n;
    let p_ws = (ws_eff << 1) | cr.pws;
    let edge_last = (ws_leads1 & (1 << 63)) | (w2 & (1 << 62)) | (w3 & (1 << 61));
    let split_ok = (ws_leads1 & (nonws >> 1))
        | (w2 & (nonws >> 2))
        | (w3 & (nonws >> 3))
        | (edge_last & nn64m);
    let mut b_ws = ws_leads & (!p_ws | split_ok);

    // Override every run containing a (non-absorbed) newline: one token through the run's last
    // newline, then the give-back rules on the remainder.
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
        let q = 63 - (nl & run_mask).leading_zeros(); // last newline in the run
        if q + 1 < e {
            // Tail after the last newline: starts a token, and its last char splits off before
            // the following non-ws char.
            b_ws |= 1u64 << (q + 1);
            let tail = run_mask & (u64::MAX << (q + 1));
            let tail_leads = ws_leads & tail;
            b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
        }
        runs_n &= !run_mask;
    }

    let mut boundary = b_letters | b_digits | b_punct | b_ws;

    // --- Contractions: `'(?i:[sdmt]|ll|ve|re)` -------------------------------------------------
    let mut cand = ap & boundary & !bad;
    while cand != 0 {
        let i = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        if i >= 61 {
            bad |= u64::MAX << i;
            break;
        }
        let b1 = text[scan + i + 1];
        if b1 >= 0x80 {
            // `(?i:'s)` also matches 'ſ (U+017F): an apostrophe before any non-ASCII char
            // defers to the scalar path.
            bad |= 0b111u64 << i;
            continue;
        }
        let k = match b1 | 0x20 {
            b's' | b'd' | b'm' | b't' => 2,
            b'l' if text[scan + i + 2] | 0x20 == b'l' => 3,
            b'v' if text[scan + i + 2] | 0x20 == b'e' => 3,
            b'r' if text[scan + i + 2] | 0x20 == b'e' => 3,
            _ => 0,
        };
        if k != 0 {
            boundary &= !(1u64 << (i + 1));
            boundary |= 1u64 << (i + k);
        }
    }

    (boundary & !bad, bad)
}
