//! The class-run family: `\w+|[^\w\s]+`, BERT's basic split, digit runs, punctuation isolation.
//! Not regex-shaped -- these cut where the atom class changes, so one shape
//! (`class_runs_into<DROP, ISOLATE, KEEP_A>`) covers all of them. Plus `CharDelimiterSplit`, the
//! only one that keys on a literal char rather than an atom class.

use crate::classify::{CONT, in_mask};
use crate::{AUX_NONE, Blk, CODE_CONT, Out, Span, blocks, build_block, emit};

/// Atom tag -> dense 2-bit code, the whole grammar: `3` DROP, `2` ISOLATE, `1` KEEP_A, `0` other.
/// The if-chain order is the precedence when the masks overlap.
const fn lut<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = if in_mask(i as u8, DROP) {
            3
        } else if in_mask(i as u8, ISOLATE) {
            2
        } else if in_mask(i as u8, KEEP_A) {
            1
        } else {
            0
        };
        i += 1;
    }
    t[CONT as usize] = CODE_CONT;
    t
}

streams!(
    /// One block's class streams — one per dense code. Only `other` needs the `valid` mask, since
    /// past the block end every plane reads 0, i.e. code 0.
    Cls { lead, cont, dropped, isolate, keep_a, other }
);

/// Decode the two planes; returns the fill seed for the block after.
fn cls(text: &[u8], tags: &[u8], base: usize, len: usize, lut: &[u8; 64], code: u8) -> (Cls, u8) {
    let valid = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
    let (Blk { p0, p1, cont, .. }, last) =
        build_block::<{ AUX_NONE }, false>(text, tags, base, len, lut, code, false);
    let c = Cls {
        lead: valid & !cont,
        cont,
        dropped: p1 & p0,
        isolate: p1 & !p0,
        keep_a: !p1 & p0,
        other: !p1 & !p0 & valid,
    };
    (c, last)
}

/// No-`push` class-family pre-tokenizer core: writes spans into the preallocated `out` slice and
/// returns the count. ONE shape covers the whole class family via `<DROP, ISOLATE, KEEP_A>`:
/// `WhitespaceSplit <{WS},0,0>` · `Punctuation <0,{PUNCT},0>` · `Digits <0,0,{NUMERIC}>` ·
/// `Whitespace <{WS},0,{WORD}>` · `Bert <{WS},{PUNCT},0>`.
/// Class of a char: `DROP`->dropped, `ISOLATE`->own token, `KEEP_A`->run "A", else->run "B"
/// (A/B cut apart). `starts` and `fake` are scratch bitmaps (len >= `text.len().div_ceil(64)`).
#[must_use]
pub fn class_runs_into<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    fake: &mut [u64],
    out: &mut [Span],
) -> usize {
    let ntext = text.len();
    if ntext == 0 {
        return 0;
    }
    let nblk = ntext.div_ceil(64);
    assert!(
        tags.len() >= ntext && starts.len() >= nblk && fake.len() >= nblk && out.len() >= ntext
    );
    let l = lut::<DROP, ISOLATE, KEEP_A>();
    blocks(
        ntext,
        &mut *starts,
        Some(fake),
        CODE_CONT,
        |base, len, seed| cls(text, tags, base, len, &l, seed),
        |x, _| {
            let (cur, bk) = (&x.cur, &x.bk);
            Out {
                // a token opens at each run start — and at every ISOLATE char, since there one char
                // is one token. A dropped run opens one too, so that whatever preceded it closes
                // there; `flag` marks it as not a token of its own.
                st: (cur.keep_a & !bk.keep_a)
                    | (cur.other & !bk.other)
                    | (cur.dropped & !bk.dropped)
                    | cur.isolate,
                patch: 0,
                flag: cur.dropped,
            }
        },
    );
    emit(starts, fake, nblk, ntext, out)
}

/// `Split(char, Removed)` — the only pre-tokenizer that keys on a *literal char* rather than an atom
/// class, so it scans bytes directly (no classify pass). UTF-8 is self-synchronizing, so the
/// delimiter's byte pattern only matches on char boundaries.
pub struct CharDelimiterSplit(pub char);
impl CharDelimiterSplit {
    /// Split on the literal char (`Removed`): writes the gaps between delimiters into `out`
    /// (len >= `text.len()`) and returns the count. Straight off [`crate::literal`] — no atom
    /// classification, UTF-8 is self-synchronising so the char's bytes only match on boundaries.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], _tags: &mut [u8], out: &mut [Span]) -> usize {
        debug_assert!(out.len() >= text.len());
        let mut buf = [0u8; 4];
        let delim = self.0.encode_utf8(&mut buf).as_bytes();
        let Ok(lit) = crate::literal::Literal::new(delim) else {
            return 0;
        };
        let (n, dl) = (text.len(), delim.len());
        let (mut start, mut w) = (0usize, 0usize);
        for m in lit.matches(text) {
            if m > start {
                out[w] = Span::new(start as u32, m as u32);
                w += 1;
            }
            start = m + dl;
        }
        if start < n {
            out[w] = Span::new(start as u32, n as u32);
            w += 1;
        }
        w
    }
}
