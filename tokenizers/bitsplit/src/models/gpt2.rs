//! **GPT-2 / ByteLevel**:
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Contractions are an ALTERNATIVE here (`don't` -> `don`, `'t`), and case-sensitive -- unlike
//! cl100k's `(?i:)`. They go to `emit_contr`'s scalar escape: a variable-length, case-optional
//! literal alternation that outranks every other arm is miserable in bit algebra and trivial there.
//!
//! The atom table and the class decode are shared with cl100k in [`super::family_gpt`].

use super::family_gpt::{cls, contractions};
use crate::{CODE_CONT, Out, Span, blocks, emit, to_lead};

/// GPT-2 / byte-level pre-tokenization. `starts` and `flag` are scratch bitmaps
/// (len ≥ `text.len().div_ceil(64)`); byte-exact with `bitsplit::fsm_byte_level`.
#[must_use]
pub fn bitsplit_byte_level(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    _flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    let ntext = text.len();
    if ntext == 0 {
        return 0;
    }
    let nblk = ntext.div_ceil(64);
    assert!(tags.len() >= ntext && starts.len() >= nblk && out.len() >= ntext);
    let mut carry = (0u64, 0u64);
    blocks(
        ntext,
        &mut *starts,
        None,
        CODE_CONT,
        |base, len, seed| cls(text, tags, base, len, seed),
        |x, _| {
            let (cur, bk) = (&x.cur, &x.bk);
            // `\s+(?!\S)` hands the run's LAST whitespace char to whatever follows: as a ` ?`
            // prefix if it is a space, else as a token of its own. Either way it opens a token —
            // unless the run ends the input, where plain `\s+` takes the lot. GPT-2 has no
            // `[\r\n]` rule, so unlike cl100k a newline is ordinary whitespace here and can be
            // the stolen char.
            let (steal, patch) = to_lead(cur.ws & x.lb & !x.eof & !x.fw.ws, cur.cont, x.pv.cont);
            // every alternative but the contraction is ` ?X+` over a class run, so a token opens
            // at each run start — pushed back one char when a literal space sits in front of it.
            let mut st = ((cur.l & !bk.l & !bk.sp)
                | (cur.n & !bk.n & !bk.sp)
                | (cur.other & !bk.other & !bk.sp)
                | (cur.ws & !bk.ws)
                | steal)
                & cur.lead;
            if x.bi == 0 {
                st |= 1;
            }
            // the contraction alternative: `'s` is its own token, so the char after it opens one
            let (opens, inner) = contractions(text, st, cur.apo, x.base, ntext, false, &mut carry);
            st = (st & !inner) | opens;
            Out { st, patch, flag: 0 }
        },
    );
    emit(starts, &[], nblk, ntext, out)
}
