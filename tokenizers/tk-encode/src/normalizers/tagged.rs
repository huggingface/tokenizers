//! Shared tag-driven normalization. Classify the input once (SIMD, via `NORM_TABLES`), copy maximal
//! **inert** runs verbatim, and hand each maximal run that a rule actually touches to `emit`. Every
//! simple normalizer (NFD/NFKD/Lowercase/StripAccents) is one call with its `active` mask + per-run op.
use std::borrow::Cow;
use std::cell::RefCell;

use atomsplit::classify::char_len;
use atomsplit::norm_classify;

thread_local! {
    /// Per-thread reused tag buffer — zero-alloc after warmup, lock-free across parallel encode threads.
    static TAGS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Classify `input`, then rebuild it: runs with no `active` bit are copied verbatim; each maximal run
/// carrying an active bit is passed to `emit(run, out)`. Returns `Cow::Borrowed` when nothing is active.
///
/// Byte-exact for any rule whose non-inert runs are self-contained. That holds for NFD/NFKD/Lowercase/
/// strip: inert chars are ccc-0 starters (reorderables carry the NFD bit), so NFD canonical reordering
/// never crosses a run boundary, and lowercase/strip are per-char. It does NOT hold for composition
/// (NFC/NFKC), which can combine across a starter — those keep their own quick-check path.
#[inline]
pub(crate) fn tag_driven<'a>(
    input: &'a str,
    active: u8,
    emit: impl Fn(&str, &mut String),
) -> Cow<'a, str> {
    if input.is_empty() {
        return Cow::Borrowed(input);
    }
    TAGS.with(|cell| {
        let mut tags = cell.borrow_mut();
        tags.clear();
        tags.resize(input.len(), 0);
        norm_classify::classify(input.as_bytes(), &mut tags);
        let bytes = input.as_bytes();
        let n = bytes.len();

        let mut first = 0;
        while first < n && tags[first] & active == 0 {
            first += char_len(bytes[first]);
        }
        if first == n {
            return Cow::Borrowed(input);
        }
        let mut out = String::with_capacity(n);
        out.push_str(&input[..first]);
        let mut i = first;
        while i < n {
            let ns = i;
            while i < n && tags[i] & active != 0 {
                i += char_len(bytes[i]);
            }
            emit(&input[ns..i], &mut out);
            let is = i;
            while i < n && tags[i] & active == 0 {
                i += char_len(bytes[i]);
            }
            out.push_str(&input[is..i]);
        }
        Cow::Owned(out)
    })
}
