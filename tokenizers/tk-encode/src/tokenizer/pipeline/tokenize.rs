//! How a model encodes words into token ids.
//!
//! A word is one pre-token: one span the pre-tokenizer cut, such as `" world"` or `"▁hello"`.
//! It need not be a word of the language.
//! Each model implements [`Tokenize`], which turns one word into ids.
//! [`tokenize_spans`] runs it over every word of a text sequence.

use crate::pipeline::{PreTokens, Token};
use crate::tokenizer::Result;
use crate::utils::cache::DEFAULT_CACHE_CAPACITY;
use crate::utils::word_cache::{MAX_INLINE_IDS, ProbeEmit, WordCache};
use crate::vocab::bucket_vocab_store;

// `probe_emit_keyed` writes ids through a `*mut u32` into the `Vec<Token>` output.
// That is sound only while a token has the same layout as u32
const _: () = assert!(size_of::<Token>() == size_of::<u32>());
const _: () = assert!(align_of::<Token>() == align_of::<u32>());

/// One word, with the key and hash the word cache and the vocabulary look it up by.
#[derive(Clone, Copy)]
pub(crate) struct Word<'a> {
    pub(crate) text: &'a str,
    pub(crate) lookup_key: u64,
    pub(crate) lookup_hash: u64,
}

/// The algorithm a model uses to turn one word into token ids.
///
/// Called by [`tokenize_spans`] when the word is not already cached.
pub(crate) trait Tokenize {
    /// The buffers the model reuses from one word to the next.
    type Scratch;

    fn new_scratch(&self) -> Self::Scratch;

    /// How many slots in the word cache or `None` for no cache
    fn word_cache_capacity(&self) -> Option<usize> {
        Some(DEFAULT_CACHE_CAPACITY)
    }

    /// Appends the ids of `word` to `output`.
    fn tokenize_word(
        &self,
        word: Word<'_>,
        scratch: &mut Self::Scratch,
        output: &mut Vec<Token>,
    ) -> Result<()>;
}

/// The word cache and the model's reused buffers
pub struct Scratch<S> {
    /// Kept across encode calls, so a word seen in one call is a hit in the next.
    pub(crate) word_cache: Option<WordCache>,
    pub(crate) model: S,
}

impl<S> Scratch<S> {
    /// An empty word cache and new buffers for `model`.
    pub(crate) fn new<M: Tokenize<Scratch = S>>(model: &M) -> Self {
        Self {
            word_cache: model.word_cache_capacity().map(WordCache::new),
            model: model.new_scratch(),
        }
    }
}

/// Appends the ids of every word of a chunk to `output`.
///
/// For each word, in order:
///  - look the word up in the word cache
///  - on a miss, call the model's [`Tokenize::tokenize_word`] and add the ids to the cache
// Inlining measured faster
#[inline(always)]
pub(crate) fn tokenize_spans<M: Tokenize>(
    model: &M,
    pre_tokens: PreTokens<'_>,
    scratch: &mut Scratch<M::Scratch>,
    output: &mut Vec<Token>,
) -> Result<()> {
    let Scratch {
        word_cache,
        model: model_scratch,
    } = scratch;

    // Turn the scratch's Option into a local pointer once, so checking if the Option is None
    // is a register compare
    let mut word_cache = word_cache.as_mut();
    let chunk = pre_tokens.chunk();
    let words = pre_tokens.spans().len();

    let mut writer = TokenWriter::new(output);

    for (span, text) in pre_tokens.iter() {
        if text.is_empty() {
            continue;
        }
        writer.make_room(words);

        let (lookup_key, lookup_hash) =
            bucket_vocab_store::lookup_key_and_hash(chunk.as_bytes(), span);

        let mut placement = None;
        if let Some(cache) = word_cache.as_deref_mut() {
            // SAFETY: `make_room` left room for `MAX_INLINE_IDS` ids at the writer's end.
            match unsafe { cache.probe_emit_keyed(lookup_key, lookup_hash, writer.end()) } {
                ProbeEmit::Wrote(written) => {
                    // SAFETY: the probe wrote `written` ids at the end.
                    unsafe { writer.advance(written) };
                    continue;
                }
                ProbeEmit::Hit(ids) => {
                    let output = writer.into_output();
                    output.extend(ids.iter().map(|&id| Token::from(id)));
                    writer = TokenWriter::new(output);
                    continue;
                }
                ProbeEmit::Miss(at) => placement = Some(at),
            }
        }

        let output = writer.into_output();
        let start = output.len();
        model.tokenize_word(
            Word {
                text,
                lookup_key,
                lookup_hash,
            },
            model_scratch,
            output,
        )?;
        if let Some(cache) = word_cache.as_deref_mut()
            && let Some(at) = placement
        {
            cache.insert(at, output[start..].iter().map(|token| token.id()));
        }
        writer = TokenWriter::new(output);
    }
    writer.into_output();
    Ok(())
}

/// Appends ids to a `Vec<Token>` through a raw pointer to its end.
///
/// A cache hit writes its ids at [`TokenWriter::end`] and counts them with
/// [`TokenWriter::advance`], which only updates the writer's own fields. Anything else that grows
/// the `Vec` takes it back with [`TokenWriter::into_output`] first, so the pointer is never stale.
// Through `Vec` methods, every hit reloaded the length and the capacity from memory.
struct TokenWriter<'a> {
    output: &'a mut Vec<Token>,
    /// How many tokens of `output` are written: its length, once given back.
    len: usize,
    capacity: usize,
    /// Where the next id goes: `len` tokens into the buffer.
    end: *mut Token,
}

impl<'a> TokenWriter<'a> {
    #[inline(always)]
    fn new(output: &'a mut Vec<Token>) -> Self {
        let len = output.len();
        let capacity = output.capacity();
        // SAFETY: `len <= capacity`, so this is inside the allocation or one past its end.
        let end = unsafe { output.as_mut_ptr().add(len) };
        Self {
            output,
            len,
            capacity,
            end,
        }
    }

    /// Makes room for [`MAX_INLINE_IDS`] ids at the end. When the buffer must grow, it grows by
    /// `words` words' worth, so the next words find room too.
    #[inline(always)]
    fn make_room(&mut self, words: usize) {
        if self.len + MAX_INLINE_IDS > self.capacity {
            // SAFETY: the first `len` tokens are written.
            unsafe { self.output.set_len(self.len) };
            self.output.reserve(words + MAX_INLINE_IDS);
            self.capacity = self.output.capacity();
            // SAFETY: `len <= capacity`; `reserve` may have moved the buffer.
            self.end = unsafe { self.output.as_mut_ptr().add(self.len) };
        }
    }

    /// Where the next id goes, as the `u32` that a [`Token`] wraps.
    #[inline(always)]
    fn end(&self) -> *mut u32 {
        self.end.cast::<u32>()
    }

    /// Counts `written` more ids as part of the output.
    ///
    /// # Safety
    ///
    /// `written` ids were written at [`Self::end`], inside the room [`Self::make_room`] made.
    #[inline(always)]
    unsafe fn advance(&mut self, written: usize) {
        self.len += written;
        // SAFETY: the caller wrote `written` ids from `end`, inside the allocation.
        self.end = unsafe { self.end.add(written) };
    }

    /// The `Vec`, with every written id counted in its length.
    #[inline(always)]
    fn into_output(self) -> &'a mut Vec<Token> {
        // SAFETY: the first `len` tokens are written.
        unsafe { self.output.set_len(self.len) };
        self.output
    }
}

/// [`tokenize_spans`] over one word: the whole of `text`.
#[cfg(test)]
pub(crate) fn tokenize_one<M: Tokenize>(
    model: &M,
    text: &str,
    scratch: &mut Scratch<M::Scratch>,
    output: &mut Vec<Token>,
) -> Result<()> {
    let spans = [crate::pipeline::Span::new(0, text.len() as u32)];
    tokenize_spans(model, PreTokens::checked(text, &spans), scratch, output)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::pipeline::{PreTokenizer, PreTokenizerScratch, Span};
    use crate::pre_tokenizers::whitespace::WhitespaceSplit;

    /// One id per byte of the word, so a word's length picks how many ids it writes.
    struct BytesModel {
        cache_capacity: Option<usize>,
        calls: Cell<usize>,
    }

    impl BytesModel {
        fn new(cache_capacity: Option<usize>) -> Self {
            Self {
                cache_capacity,
                calls: Cell::new(0),
            }
        }
    }

    impl Tokenize for BytesModel {
        type Scratch = ();

        fn new_scratch(&self) {}

        fn word_cache_capacity(&self) -> Option<usize> {
            self.cache_capacity
        }

        fn tokenize_word(&self, word: Word<'_>, _: &mut (), output: &mut Vec<Token>) -> Result<()> {
            self.calls.set(self.calls.get() + 1);
            output.extend(word.text.bytes().map(|byte| Token::from(byte as u32)));
            Ok(())
        }
    }

    /// The ids of every space-separated word of `text`, appended after an id already in the
    /// output.
    fn ids(model: &BytesModel, text: &str) -> Vec<u32> {
        let mut spans = Vec::new();
        let pre_tokens = PreTokenizer::WhitespaceSplit(WhitespaceSplit)
            .pre_tokens(text, &mut PreTokenizerScratch::default(), &mut spans)
            .unwrap();
        let mut scratch = Scratch::new(model);
        let mut output = vec![Token::from(999)];
        tokenize_spans(model, pre_tokens, &mut scratch, &mut output).unwrap();
        output.iter().map(|token| token.id()).collect()
    }

    // Words of 1, 3 (inline in a slot) and 6 ids (spilled), each seen many times, so hits and
    // misses alternate and the output grows past its first reservation.
    #[test]
    fn hits_and_misses_in_one_chunk_give_the_ids_of_an_uncached_run() {
        let text = "a bcd efghij ".repeat(200);

        let cached = ids(&BytesModel::new(Some(1024)), &text);
        let uncached = ids(&BytesModel::new(None), &text);

        assert_eq!(cached, uncached);
        assert_eq!(cached.len(), 1 + 200 * 10);
    }

    #[test]
    fn a_repeated_word_runs_the_model_once() {
        let model = BytesModel::new(Some(1024));

        ids(&model, "a bcd efghij a bcd efghij");

        assert_eq!(model.calls.get(), 3);
    }

    // Without a cache, so no slot can answer the empty word first.
    #[test]
    fn an_empty_word_never_reaches_the_model() {
        let model = BytesModel::new(None);
        let spans = [Span::new(0, 0), Span::new(0, 2), Span::new(2, 2)];
        let pre_tokens = PreTokens::checked("ab", &spans);
        let mut output = Vec::new();

        tokenize_spans(&model, pre_tokens, &mut Scratch::new(&model), &mut output).unwrap();

        let ids: Vec<u32> = output.iter().map(|token| token.id()).collect();
        assert_eq!(ids, [97, 98]);
        assert_eq!(model.calls.get(), 1);
    }
}
