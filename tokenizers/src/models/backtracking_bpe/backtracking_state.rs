use super::bitfield::BitField;

/// This can be thought of as a lazy variation of the dynamic programming approach.
/// It only computes those states which have to be visited in order to compute the tokenization
/// for a given input text.
/// It keeps track of visited states in a bitfield and only remembers the tokenization
/// of the currently processed dynamic programming state.
///
/// The biggest downside of this approach is that the search for the longest leftmost match (the firt token?)
/// has to be reset at every (backtracking) step which is still a net win in practice compared to other approaches.
#[derive(Clone, PartialEq)]
pub struct BacktrackState<'a> {
    pub(crate) text: &'a [u8],
    pub(crate) tokens: Vec<u32>,        // len of the tezt / 3
    pub(crate) next_token: Option<u32>, // bpe.next_match(text) wich is longest_searcher.leftmost_find_iter(text)'s first match value
    pub(crate) pos: usize,              // current pos in the text?
    pub(crate) bitfield: BitField, // keeps track of token boundaries? keeps track of all the valid tokenization positions and making the runtime linear in the input length.
}

impl<'a> BacktrackState<'a> {
    pub(crate) fn new(text: &'a [u8], next_token: Option<u32>) -> Self {
        Self::with_capacity(text, next_token, text.len() / 3)
    }

    pub(crate) fn with_capacity(text: &'a [u8], next_token: Option<u32>, cap: usize) -> Self {
        Self {
            text,
            tokens: Vec::with_capacity(cap),
            next_token,
            pos: 0,
            bitfield: BitField::new(text.len() + 1),
        }
    }
    pub(crate) fn count(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn last_token(&self) -> Option<u32> {
        self.tokens.last().copied()
    }

    pub(crate) fn into_tokens(self) -> Vec<u32> {
        self.tokens
    }
}
