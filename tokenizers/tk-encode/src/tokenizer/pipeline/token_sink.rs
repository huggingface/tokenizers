//! Where a model writes its tokens.

use super::PipelineToken;

/// Somewhere tokens are written: a growable [`Vec`], or a fixed slot in a batch's buffer.
///
/// The models write through this so the same code can fill a `Vec` or write straight into one
/// document's slot of a pre-allocated batch, with no copy in between.
///
/// A slot can run out. When it does, [`Self::room`] returns `None` and every later write is
/// dropped -- the model is not asked to unwind, because the caller re-encodes that one document
/// into a `Vec` instead. So an implementation only has to stay memory-safe past the end, not
/// correct; correctness comes from the re-encode.
pub trait TokenSink: Extend<PipelineToken> {
    /// A cursor with space for `n` more tokens, or `None` when there is no room for them.
    fn room(&mut self, n: usize) -> Option<*mut PipelineToken>;

    /// The tokens written so far.
    fn written(&self) -> &[PipelineToken];

    /// Count `n` more tokens as written.
    ///
    /// # Safety
    ///
    /// `n` tokens must have been written at the cursor the matching [`Self::room`] returned, and
    /// `n` must not exceed what was asked for.
    unsafe fn advance(&mut self, n: usize);

    /// Drop everything past `len`.
    fn truncate(&mut self, len: usize);

    fn len(&self) -> usize {
        self.written().len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push(&mut self, token: PipelineToken) {
        if let Some(at) = self.room(1) {
            // SAFETY: `room` returned a cursor with space for one token, which is what is written.
            unsafe {
                at.write(token);
                self.advance(1);
            }
        }
    }
}

impl TokenSink for Vec<PipelineToken> {
    fn room(&mut self, n: usize) -> Option<*mut PipelineToken> {
        self.reserve(n);
        // SAFETY: `reserve` leaves at least `n` slots past `len`, inside the allocation.
        Some(unsafe { self.as_mut_ptr().add(self.len()) })
    }

    fn written(&self) -> &[PipelineToken] {
        self
    }

    unsafe fn advance(&mut self, n: usize) {
        // SAFETY: the caller wrote `n` tokens into the slots `room` reserved.
        unsafe { self.set_len(self.len() + n) };
    }

    fn truncate(&mut self, len: usize) {
        self.truncate(len);
    }
}

/// One document's slot in a batch buffer: a fixed run of tokens that cannot grow.
///
/// Writes past the end are dropped and remembered in [`Self::overflowed`], which is the caller's
/// signal to re-encode this document into a `Vec` and spill it.
pub struct TokenSlot<'a> {
    slot: &'a mut [PipelineToken],
    len: usize,
    overflowed: bool,
}

impl<'a> TokenSlot<'a> {
    pub fn new(slot: &'a mut [PipelineToken]) -> Self {
        Self {
            slot,
            len: 0,
            overflowed: false,
        }
    }

    /// Whether a write did not fit, so what this holds is incomplete.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }
}

impl Extend<PipelineToken> for TokenSlot<'_> {
    fn extend<I: IntoIterator<Item = PipelineToken>>(&mut self, tokens: I) {
        for token in tokens {
            self.push(token);
        }
    }
}

impl TokenSink for TokenSlot<'_> {
    fn room(&mut self, n: usize) -> Option<*mut PipelineToken> {
        if self.len + n > self.slot.len() {
            self.overflowed = true;
            return None;
        }
        // SAFETY: the bound above puts `len + n` inside the slot.
        Some(unsafe { self.slot.as_mut_ptr().add(self.len) })
    }

    fn written(&self) -> &[PipelineToken] {
        &self.slot[..self.len]
    }

    unsafe fn advance(&mut self, n: usize) {
        self.len += n;
    }

    fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(id: u32) -> PipelineToken {
        PipelineToken::from(id)
    }

    #[test]
    fn a_vec_grows_and_a_slot_does_not() {
        let mut vec: Vec<PipelineToken> = Vec::new();
        vec.extend((0..100).map(token));
        assert_eq!(vec.len(), 100);

        let mut backing = vec![token(0); 3];
        let mut slot = TokenSlot::new(&mut backing);
        slot.extend((0..2).map(token));
        assert!(!slot.overflowed());
        assert_eq!(slot.written(), [token(0), token(1)]);

        // The third fits, the fourth does not, and is dropped rather than written past the end.
        slot.push(token(2));
        slot.push(token(3));
        assert!(slot.overflowed());
        assert_eq!(slot.written(), [token(0), token(1), token(2)]);
    }

    #[test]
    fn truncate_backtracks_like_wordpiece() {
        let mut backing = vec![token(0); 8];
        let mut slot = TokenSlot::new(&mut backing);
        slot.extend((0..5).map(token));
        let checkpoint = 2;
        slot.truncate(checkpoint);
        slot.push(token(99));
        assert_eq!(slot.written(), [token(0), token(1), token(99)]);
    }
}
