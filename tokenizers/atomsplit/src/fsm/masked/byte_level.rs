//! Masked scheme for the byte-level (GPT-2) regex — the r50k boundary algebra (via gigatoken's
//! `r50k.rs`, MIT) over tag-fed class masks.
//!
//! Every byte-level rule is local: a token starts exactly at a class change that is not a space
//! prefix, at the first byte of a whitespace run, at the last whitespace byte before a
//! non-whitespace (the `\s+(?!\S)` give-back), or at a contraction edge. Bad zones: a
//! whitespace char straddling the batch edge (its give-back is per char, not per byte), an
//! apostrophe too close to the edge for the contraction peek, a multi-byte whitespace char, or
//! a `Sentinel`/`MultiByte` tag.

use super::super::byte_level::advance_byte_level;
use super::block::Block;
use super::{MaskedFsm, char_lead, cont_runs, fill};
use crate::fsm::{APO, CONT, LET, NLN, NO, NW, SPC, WSO, in_mask, mask};

pub(super) struct ByteLevelMasked;

impl MaskedFsm for ByteLevelMasked {
    #[inline(always)]
    fn batch_masks(&self, text: &[u8], tags: &[u8], scan: usize) -> (u64, u64) {
        batch_masks(text, tags, scan)
    }

    #[inline(always)]
    fn advance(&self, text: &[u8], tags: &[u8], i: usize, end: usize) -> usize {
        advance_byte_level(text, tags, i, end)
    }
}

#[inline(always)]
fn batch_masks(text: &[u8], tags: &[u8], scan: usize) -> (u64, u64) {
    debug_assert!(scan + 64 < tags.len() && tags.len() == text.len());
    // SAFETY: `scan + 64 < tags.len()` (walker guarantee), the block's load contract.
    let b = unsafe { Block::load(tags, scan) };
    if b.any_range_tag(13, 1) {
        // Sentinel / MultiByte: the scalar dispatch has a defensive arm for these; keep its
        // behavior by refusing the whole batch.
        return (0, u64::MAX);
    }
    let l0 = b.eq_tag(LET);
    let d0 = b.range_tag(NW, 1);
    let ws0 = b.range_tag(NLN, 2);
    let s = b.eq_tag(SPC);
    // Apostrophes and continuations only matter to the fixups below; skip their movemask when
    // an any-test says the batch has none.
    let ap = if b.any_eq_full(APO) {
        b.eq_full(APO)
    } else {
        0
    };
    let c = if b.any_eq_full(CONT) {
        b.eq_full(CONT)
    } else {
        0
    };
    let (mut l, mut d, mut ws) = (l0, d0, ws0);

    // Carries: the classes of the byte just before the batch, which is the class of the char
    // containing it.
    let (pl, pd, pws, ps, po) = if scan == 0 {
        (0u64, 0u64, 0u64, 0u64, 0u64)
    } else {
        match tags[char_lead(tags, scan - 1)] & 0x0F {
            LET => (1, 0, 0, 0, 0),
            NW | NO => (0, 1, 0, 0, 0),
            SPC => (0, 0, 1, 1, 0),
            NLN | WSO => (0, 0, 1, 0, 0),
            _ => (0, 0, 0, 0, 1),
        }
    };

    if c != 0 {
        // A char straddling into the batch has its leading continuation bytes take the carry
        // class ("other" needs no action: it is derived as the complement).
        if c & 1 != 0 {
            let lead_in = c & ((1u64 << (!c).trailing_zeros()) - 1);
            l |= lead_in * pl;
            d |= lead_in * pd;
            ws |= lead_in * pws;
        }
        let (c2, c3) = cont_runs(c);
        l = fill(l, c, c2, c3);
        d = fill(d, c, c2, c3);
        ws = fill(ws, c, c2, c3);
        if ws & c != 0 {
            // Multi-byte whitespace: the `\s+(?!\S)` give-back is one char, not one byte, and
            // the algebra below works in bytes. Rare (NBSP and friends); scalar batch.
            return (0, u64::MAX);
        }
    }
    let o = !(l | d | ws);

    // The r50k boundary algebra (gigatoken, MIT). A byte starts a token when it is not
    // whitespace, does not continue a same-class run, and does not follow a space (the ` ?`
    // prefix glues it to the space instead).
    let cont_same = (l & ((l << 1) | pl)) | (d & ((d << 1) | pd)) | (o & ((o << 1) | po));
    let after_sp = (s << 1) | ps;
    let nb = !ws & !cont_same & !after_sp;

    let mut bad = 0u64;

    // Whitespace-run splits. `split_ok` = the last whitespace byte before a non-whitespace (the
    // `\s+(?!\S)` give-back starts a token there); bit 63 needs the lookahead tag.
    let mut split_ok = ws & (!ws >> 1);
    let la = tags[scan + 64] & 0x0F;
    if la == CONT {
        // The char at the batch edge straddles out. Whitespace is the only class whose rules
        // look at char ends, so only a whitespace lead poisons the tail.
        let p = char_lead(tags, scan + 63);
        if in_mask(tags[p], mask::WS) {
            bad |= u64::MAX << (p - scan);
        }
    } else if !in_mask(la, mask::WS) {
        split_ok |= ws & (1u64 << 63);
    }
    let pwsb = (ws << 1) | pws;
    let wsboundary = ws & (!pwsb | split_ok);
    let mut boundary = nb | wsboundary;

    // Contraction fixup, only when the batch has an apostrophe that starts a token: a match
    // (case-sensitive, as in the scalar dispatch) absorbs the next 1-2 letters and re-opens a
    // token after them. Too close to the edge (i >= 61) the peek and the re-opened bit can
    // cross the batch; refuse the tail instead.
    if ap != 0 {
        let mut cand = ap & boundary;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i >= 61 {
                bad |= u64::MAX << i;
                break;
            }
            let k = match text[scan + i + 1] {
                b's' | b't' | b'm' | b'd' => 2,
                b'r' | b'v' if text[scan + i + 2] == b'e' => 3,
                b'l' if text[scan + i + 2] == b'l' => 3,
                _ => 0,
            };
            if k != 0 {
                boundary &= !(1u64 << (i + 1));
                boundary |= 1u64 << (i + k);
            }
        }
    }
    (boundary & !bad, bad)
}
