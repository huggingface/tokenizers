use std::{
    mem,
    sync::{Mutex, PoisonError},
};

use atomsplit::fsm::Span;

use crate::pipeline::{Model, PipelineModel, PipelineModelScratch};

pub trait ModelScratch {}

#[derive(Default)]
pub(crate) struct EncodeScratch {
    pub(crate) model: PipelineModelScratch,
    pub(crate) pre_tokens: Vec<Span>,
}

/// A pool of [`PipelineModelScratch`].
///
/// When calling [`PipelineTokenizer::encode`], an instance of [`PipelineModelScratch`] is taken out of this pool
/// and given to the tokenizer. When the encoding is done, the scratch buffer is returned to the pool and can be
/// reused by later calls.
///
/// The reusability matters because the scratch buffer may hold cache structures which are more useful when reused,
/// and less importantly it saves an extra allocation for an fresh buffer every time.
pub(super) struct ScratchPool(Mutex<Vec<EncodeScratch>>);

impl ScratchPool {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// Get a scratch buffer from the pool, wrapped in a [`ScratchGuard`].
    /// When the [`ScratchGuard`] gets dropped, the scratch buffer is pushed back to the pool.
    pub(crate) fn get<'a>(&'a self, model: &PipelineModel) -> ScratchGuard<'a> {
        // The Mutex lock is held just long enough to pop the scratch out of the pool
        let taken = self.0.lock().unwrap_or_else(PoisonError::into_inner).pop();
        ScratchGuard {
            // If there was no scratch buffer available in the pool, we build.a fresh one
            scratch: taken.unwrap_or_else(|| EncodeScratch {
                model: model.init_scratch(),
                pre_tokens: Vec::new(),
            }),
            pool: self,
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

/// A wrapper around [`PipelineModelScratch`].
/// Implements [`Deref`] and [`DerefMut`], so it behaves as [`PipelineModelScratch`].
///
/// When it gets dropped, it pushes [`Self::scratch`] back into the shared [`Self::pool`] so it can
/// get reused by a later call to [`PipelineTokenizer::encode`].
///
/// TODO @McPatate : The Mutex can create contention, to be replaced by a better access pattern
pub(super) struct ScratchGuard<'a> {
    scratch: EncodeScratch,
    pool: &'a ScratchPool,
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        // Steals the scratch buffer from self, replaces it with PipelineModelScratch::default()
        let scratch = mem::take(&mut self.scratch);
        // Push the scratch back in the pool
        self.pool
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    }
}

impl std::ops::Deref for ScratchGuard<'_> {
    type Target = EncodeScratch;
    fn deref(&self) -> &EncodeScratch {
        &self.scratch
    }
}

impl std::ops::DerefMut for ScratchGuard<'_> {
    fn deref_mut(&mut self) -> &mut EncodeScratch {
        &mut self.scratch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;
    use crate::pipeline::PipelineTokenizer;

    /// A BPE tokenizer that merges "hello" into the single id 7.
    fn hello_tokenizer() -> Tokenizer {
        use crate::models::bpe::{BpeBuilder, Merges, Vocab};

        let vocab: Vocab = [
            ("h", 0u32),
            ("e", 1),
            ("l", 2),
            ("o", 3),
            ("he", 4),
            ("hel", 5),
            ("hell", 6),
            ("hello", 7),
        ]
        .into_iter()
        .map(|(s, i)| (s.to_string(), i))
        .collect();
        let merges: Merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "l".to_string()),
            ("hell".to_string(), "o".to_string()),
        ];
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        Tokenizer::new(bpe)
    }

    fn hello_pipeline() -> PipelineTokenizer {
        PipelineTokenizer::try_from(&hello_tokenizer()).unwrap()
    }

    // The pool exists so ONE `&self` tokenizer can be shared across rayon workers. Encode
    // the same input from thousands of threads through a single instance; each must get a
    // private scratch and produce the sequential result. Two threads sharing a scratch
    // would corrupt some of them. This only compiles if `PipelineTokenizer: Sync`,
    // which the pool has to preserve.
    #[test]
    fn encode_shared_across_threads() {
        use rayon::prelude::*;

        let pipeline = hello_pipeline();

        let want: Vec<u32> = pipeline
            .encode("hello", false)
            .wait()
            .unwrap()
            .remove(0)
            .iter()
            .map(|t| t.id())
            .collect();
        assert_eq!(want, vec![7]);

        let all_match = (0..10_000u32).into_par_iter().all(|_| {
            pipeline
                .encode("hello", false)
                .wait()
                .unwrap()
                .remove(0)
                .iter()
                .map(|t| t.id())
                .collect::<Vec<_>>()
                == want
        });
        assert!(all_match);
    }

    // Reusing scratches is the whole point of the pool, so it must not build one per call:
    // one thread encoding in a loop has to keep coming back to the same scratch, and a
    // burst of N threads must leave at most N behind for later calls to use.
    #[test]
    fn scratches_are_reused_rather_than_piling_up() {
        use std::sync::Barrier;

        let pipeline = hello_pipeline();
        for _ in 0..1000 {
            pipeline.encode("hello", false).wait().unwrap();
        }
        assert_eq!(pipeline.scratch_pool.len(), 1);

        let threads = 64;
        let all_holding = Barrier::new(threads);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let scratch = pipeline.scratch_pool.get(&pipeline.model);
                    all_holding.wait();
                    drop(scratch);
                });
            }
        });

        let after_burst = pipeline.scratch_pool.len();
        assert!(
            after_burst <= threads,
            "{after_burst} scratches kept for {threads} threads"
        );
        for _ in 0..1000 {
            pipeline.encode("hello", false).wait().unwrap();
        }
        assert_eq!(pipeline.scratch_pool.len(), after_burst);
    }

    // ScratchGuard::drop moves the scratch to the pool with mem::take, which leaves a default
    // EncodeScratch, holding the None model variant, behind in the guard. The pool must end up
    // holding the scratch the model used: if the default leftover were pushed instead, the pool
    // would still count one scratch, and the next encode would take it and panic in
    // PipelineModel::tokenize_pipeline.
    #[test]
    fn the_pool_gets_back_the_used_scratch_not_the_none_leftover() {
        let pipeline = hello_pipeline();
        pipeline.encode("hello", false).wait().unwrap();
        assert_eq!(pipeline.scratch_pool.len(), 1);
        let scratch = pipeline.scratch_pool.get(&pipeline.model);
        assert!(
            matches!(scratch.model, PipelineModelScratch::BPE(_)),
            "the pooled scratch is not the BPE scratch the model used"
        );
    }

    // Pooling is only worth it if a scratch keeps the state it built up across calls.
    // A fresh BpeScratch starts with an empty merge arena; encoding a longer sequence
    // reserves one entry per byte and so grows it. After a second, short encode the
    // pooled scratch must still be the grown one, not a fresh replacement built
    // somewhere along the way.
    #[test]
    fn a_reused_scratch_keeps_its_grown_buffers() {
        let pipeline = hello_pipeline();
        let long_input = "hello".repeat(50);
        pipeline.encode(&long_input, false).wait().unwrap();
        assert_eq!(pipeline.scratch_pool.len(), 1);

        pipeline.encode("hello", false).wait().unwrap();
        assert_eq!(pipeline.scratch_pool.len(), 1);

        let scratch = pipeline.scratch_pool.get(&pipeline.model);
        let PipelineModelScratch::BPE(bpe_scratch) = &scratch.model else {
            panic!("the pooled scratch is not a BPE scratch");
        };
        assert!(
            bpe_scratch.queue.entries.capacity() >= long_input.len(),
            "the pooled scratch is not the one grown by the long input: \
             room for {} merge entries after a {}-byte input",
            bpe_scratch.queue.entries.capacity(),
            long_input.len()
        );
    }

    // The pre-token spans travel in the same EncodeScratch as the model state, for the same
    // reason: the buffer a call grows is the buffer the next call writes into. The tokenizer
    // gets a Whitespace pre-tokenizer here so the input is cut into many spans, where the
    // pipelines above leave the whole chunk as one.
    #[test]
    fn a_reused_scratch_keeps_its_pre_token_buffer() {
        use crate::pre_tokenizers::whitespace::Whitespace;

        let mut tok = hello_tokenizer();
        tok.with_pre_tokenizer(Some(Whitespace));
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();

        let words = 50;
        pipeline
            .encode("hello ".repeat(words).as_str(), false)
            .wait()
            .unwrap();

        let scratch = pipeline.scratch_pool.get(&pipeline.model);
        assert!(
            scratch.pre_tokens.capacity() >= words,
            "the pooled pre-token buffer is not the one grown by the {words} words: \
             room for {} spans",
            scratch.pre_tokens.capacity()
        );
    }

    // A scratch coming back out of the pool has to still know the words of the last
    // encode: a cache emptied between calls would never hit.
    //
    // "helo" and not "hello": a whole word that is itself a vocabulary entry is answered
    // by the fold in one probe, before the cache is ever consulted, so it would never be
    // stored. Only words that reach the merge engines land in the cache.
    #[test]
    fn the_word_cache_outlives_the_encode_call() {
        let pipeline = hello_pipeline();
        pipeline.encode("helo", false).wait().unwrap();

        let mut scratch = pipeline.scratch_pool.get(&pipeline.model);
        let PipelineModelScratch::BPE(bpe) = &mut scratch.model else {
            panic!("a BPE pipeline encodes with a BPE scratch");
        };
        let cache = bpe.word_cache.as_mut().expect("BPE encodes with a cache");
        assert_eq!(cache.lookup(b"helo").hit(), Some(&[5u32, 3][..]));
    }
}
