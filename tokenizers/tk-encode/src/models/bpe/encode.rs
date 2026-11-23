//! How a BPE turns the pre-tokens of a chunk into tokens: the word cache first, then the
//! vocabulary, then the merge engines of [`BPE::merge_word`].
use crate::models::bpe::At;
use crate::models::bpe::merge_hot_cold_queue::QueueScratch;
use crate::models::bpe::model::BPE;
use crate::pipeline::{self, PreTokens, Token};
use crate::tokenizer::Result;
use crate::utils::signature::Signature;
use crate::utils::word_cache::{Lookup, WordCache};

/// How many words `tokenize_spans` reads the signatures of before it looks any of them up.
// Measured at 8, 16, 32 and 64 on an EPYC 7R13: each step up ran fewer instructions per word, and
// 32 and 64 were the fastest, on text the cache had seen and on text it had not.
const BATCH: usize = 64;

/// Reads the signature of every word of `batch` into `signatures`, and starts loading each
/// word's home slot in the cache.
///
/// A signature costs the same whatever the word's length, with no branch. So every slot is on its
/// way to the CPU cache before the first lookup needs one.
// Out of line: inside the encode loop, the compiler reloaded the loop's addresses per word.
#[inline(never)]
fn read_signatures(batch: PreTokens<'_>, signatures: &mut [Signature], cache: Option<&WordCache>) {
    let chunk = batch.chunk().as_bytes();
    let words = signatures.iter_mut().zip(batch.spans());
    // One loop per case, so that neither tests for a cache on every word.
    match cache {
        Some(cache) => {
            for (signature, span) in words {
                // SAFETY: every span of a `PreTokens` is a range of its chunk.
                *signature = unsafe { Signature::new_in(chunk, span.range()) };
                cache.prefetch(*signature);
            }
        }
        None => {
            for (signature, span) in words {
                // SAFETY: every span of a `PreTokens` is a range of its chunk.
                *signature = unsafe { Signature::new_in(chunk, span.range()) };
            }
        }
    }
}

impl BPE {
    #[inline(always)]
    fn fold_id(&self, word: &[u8], signature: Signature) -> Option<u32> {
        // 99% of tokens fit a signature, and a word that does not is rarely a token. Merging it
        // gives the same ids, without hashing all of its bytes for a probe that misses.
        if signature.len > Signature::BYTES && !self.long_folds.may_hold(signature) {
            return None;
        }
        // One probe; the foldable bit is part of the id that probe already returned. Which entries
        // carry it was settled at load -- see `from_merge_map`.
        let (id, foldable) = self.vocab.get_foldable(word, signature)?;
        foldable.then_some(id)
    }

    /// Tokenizes one word whose signature is already read: from the cache, else as its own
    /// vocabulary entry, else through the merge engine.
    ///
    /// The cache goes first. It is one load, where the fold is a perfect-hash probe into the
    /// whole vocabulary. The two agree on ids: `prove_fold` only sets the bit for an entry that
    /// merging its own text reproduces. A folded word is inserted too, so its later occurrences
    /// come off the cache.
    // Out of line: it holds the lookup, the fold and the merge call, inlined, and has two callers.
    #[inline(never)]
    fn tokenize_with_signature(
        &self,
        sequence: &str,
        signature: Signature,
        scratch: &mut BpeScratch,
        output: &mut Vec<Token>,
    ) {
        if sequence.is_empty() {
            return;
        }
        let bytes = sequence.as_bytes();
        let BpeScratch {
            symbols,
            queue,
            word_cache,
        } = scratch;

        let mut insert_at = None;
        if let Some(cache) = word_cache.as_ref() {
            match cache.lookup_with_signature(bytes, signature) {
                Lookup::Hit(ids) => {
                    ids.append_to(output);
                    return;
                }
                Lookup::Miss(at) => insert_at = Some(at),
            }
        }

        let start = output.len();
        match self.fold_id(bytes, signature) {
            Some(id) => output.push(Token::from(id)),
            None => {
                self.merge_word(sequence, symbols, queue);
                // The merge engines work in internal ids; `unmap` takes them back to the vocab's own ids.
                output.extend(
                    symbols
                        .iter()
                        .map(|&symbol| Token::from(self.tables.unmap.at(symbol as usize))),
                );
            }
        }
        if let (Some(cache), Some(at)) = (word_cache.as_mut(), insert_at) {
            cache.insert(at, bytes, output[start..].iter().map(|token| token.id()));
        }
    }

    /// Tokenizes a batch of words whose signatures are already read.
    ///
    /// [`WordCache::emit_home_hits`] answers runs of words straight from the cache, and
    /// [`Self::tokenize_with_signature`] takes each word that a run stops on.
    // Out of line, with the run loop inlined into it: on text the cache has not seen a run is
    // 2 to 5 words long, and a call per run cost more than the run.
    #[inline(never)]
    fn tokenize_batch(
        &self,
        batch: PreTokens<'_>,
        signatures: &[Signature],
        scratch: &mut BpeScratch,
        output: &mut Vec<Token>,
    ) {
        let mut next = 0;
        while next < signatures.len() {
            if let Some(cache) = scratch.word_cache.as_ref() {
                let word = |i| batch.text(next + i).as_bytes();
                next += cache.emit_home_hits(&signatures[next..], word, output);
                if next == signatures.len() {
                    break;
                }
            }
            self.tokenize_with_signature(batch.text(next), signatures[next], scratch, output);
            next += 1;
        }
    }
}

/// Per-thread scratch for BPE. Every buffer here is cleared, never reallocated, so tokenizing a
/// sequence does not allocate.
pub struct BpeScratch {
    /// Symbols of the word being merged.
    pub(crate) symbols: Vec<u32>,
    /// Entry arena and the two queue tiers.
    pub(crate) queue: QueueScratch,
    /// Words already seen, so a repeat costs a probe instead of a merge. It lives in the scratch
    /// so it outlives the encode call that fills it -- otherwise it would never see a word twice.
    pub(crate) word_cache: Option<WordCache>,
}

impl pipeline::Tokenize for BPE {
    type Scratch = BpeScratch;

    fn tokenize(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<Token>,
    ) -> Result<()> {
        let signature = Signature::new(sequence.as_bytes());
        self.tokenize_with_signature(sequence, signature, scratch, output);
        Ok(())
    }

    /// Every pre-token of a chunk in one call, [`BATCH`] words at a time.
    ///
    /// For each batch, [`read_signatures`] reads every word's signature first, then
    /// [`BPE::tokenize_batch`] looks the words up.
    // Inlining measure faster on x86
    #[inline(always)]
    fn tokenize_spans(
        &self,
        pre_tokens: PreTokens<'_>,
        scratch: &mut Self::Scratch,
        output: &mut Vec<Token>,
    ) -> Result<()> {
        let mut signatures = [Signature::default(); BATCH];
        for batch in pre_tokens.batches(BATCH) {
            let signatures = &mut signatures[..batch.spans().len()];
            read_signatures(batch, signatures, scratch.word_cache.as_ref());
            self.tokenize_batch(batch, signatures, scratch, output);
        }
        Ok(())
    }

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            symbols: Vec::with_capacity(64),
            queue: QueueScratch::default(),
            word_cache: self.cache_capacity.map(WordCache::new),
        }
    }
}
