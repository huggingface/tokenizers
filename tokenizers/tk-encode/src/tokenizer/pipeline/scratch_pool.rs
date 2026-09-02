use std::{
    mem,
    sync::{Mutex, PoisonError},
};

use bitsplit::Span;

use crate::pipeline::{Model, PipelineModel, PipelineModelScratch, PreTokenizerScratch};

pub trait ModelScratch {}

#[derive(Default)]
pub(crate) struct EncodeScratch {
    pub(crate) model: PipelineModelScratch,
    pub(crate) split: PreTokenizerScratch,
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
        self.get_with(|| EncodeScratch {
            model: model.init_scratch(),
            split: PreTokenizerScratch::default(),
            pre_tokens: Vec::new(),
        })
    }

    pub(crate) fn get_with<'a>(&'a self, init: impl FnOnce() -> EncodeScratch) -> ScratchGuard<'a> {
        // The Mutex lock is held just long enough to pop the scratch out of the pool
        let taken = self.0.lock().unwrap_or_else(PoisonError::into_inner).pop();
        ScratchGuard {
            scratch: taken.unwrap_or_else(init),
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
    use std::cell::Cell;
    use std::collections::BTreeSet;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::models::bpe::PipelineBPE;
    use crate::pipeline::{
        PipelineModel, PipelinePostProcessor, PipelinePreTokenizer, PipelineTokenizer,
    };
    use crate::pre_tokenizers::sequence::PipelineSequence;
    use crate::pre_tokenizers::whitespace::Whitespace;
    use crate::vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary;

    /// A BPE model that merges "hello" into the single id 7.
    fn hello_bpe() -> PipelineBPE {
        use crate::models::bpe::{BpeConfig, Merges, Vocab};

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
        PipelineBPE::from_config(BpeConfig {
            vocab,
            merges,
            ..BpeConfig::default()
        })
        .unwrap()
    }

    /// Assembled through `from_parts` rather than from a `Tokenizer`: the config layer lives in
    /// `tk-convert` now, and the scratch pool does not care which reader filled the parts.
    fn hello_pipeline_with(pre_tokenizer: PipelinePreTokenizer) -> PipelineTokenizer {
        PipelineTokenizer::from_parts(
            BucketAddedVocabulary::new(),
            Vec::new(),
            pre_tokenizer,
            PipelineModel::BPE(hello_bpe()),
            PipelinePostProcessor::default(),
            None,
            Default::default(),
            None,
        )
    }

    fn hello_pipeline() -> PipelineTokenizer {
        hello_pipeline_with(PipelinePreTokenizer::None)
    }

    /// The same model, with a pre-tokenizer that cuts one span per word. A pipeline with no
    /// pre-tokenizer leaves the whole chunk as a single span, which barely exercises
    /// [`EncodeScratch::pre_tokens`].
    fn hello_pipeline_split_on_words() -> PipelineTokenizer {
        hello_pipeline_with(PipelinePreTokenizer::Whitespace(Whitespace))
    }

    /// The same model, with a `Sequence` of two real children: the shape that runs
    /// [`PipelineSequence`]'s child loop, where a single child or a recognized deepseek
    /// composition would take a fast path above it instead.
    fn hello_pipeline_sequence() -> PipelineTokenizer {
        use crate::pre_tokenizers::digits::Digits;
        use crate::pre_tokenizers::whitespace::WhitespaceSplit;

        hello_pipeline_with(PipelinePreTokenizer::Sequence(PipelineSequence::new(vec![
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelinePreTokenizer::Digits(Digits::default()),
        ])))
    }

    /// Builds a default scratch and counts the call, for the tests that assert the pool reuses
    /// what it has instead of building.
    fn counting_init(built: &AtomicUsize) -> EncodeScratch {
        built.fetch_add(1, Ordering::Relaxed);
        EncodeScratch::default()
    }

    // Reusing scratches is the whole point of the pool, so one caller coming back a thousand
    // times must keep getting the one scratch the first call built.
    #[test]
    fn builds_one_scratch_and_reuses_it() {
        let pool = ScratchPool::new();
        let built = Cell::new(0);

        for _ in 0..1000 {
            drop(pool.get_with(|| {
                built.set(built.get() + 1);
                EncodeScratch::default()
            }));
        }

        assert_eq!(built.get(), 1);
    }

    // N callers holding at once need N scratches, and the pool must keep no more than that: it is
    // there to hand scratches back out, not to grow. The barrier makes all N hold before any of
    // them drops, so all N really are live at the same time.
    #[test]
    fn keeps_at_most_one_scratch_per_concurrent_holder() {
        let pool = ScratchPool::new();
        let built = AtomicUsize::new(0);

        let threads = 64;
        let all_holding = Barrier::new(threads);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let scratch = pool.get_with(|| counting_init(&built));
                    all_holding.wait();
                    drop(scratch);
                });
            }
        });
        let after_burst = pool.len();
        assert!(
            after_burst <= threads,
            "{after_burst} scratches kept for {threads} threads"
        );

        drop(pool.get_with(|| counting_init(&built)));
        assert_eq!(
            built.load(Ordering::Relaxed),
            threads,
            "the pool built a fresh scratch while it had {after_burst} to hand out"
        );
    }

    // `Drop` runs while a panic unwinds, so an encode that blows up half way through still hands
    // its scratch back rather than taking it with it.
    //
    // The panic message on stderr during this test is the one raised here on purpose.
    #[test]
    fn a_panicking_holder_still_returns_its_scratch() {
        let pool = ScratchPool::new();
        let built = AtomicUsize::new(0);

        let holder = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scratch = pool.get_with(|| counting_init(&built));
            panic!("encode blew up while holding a scratch");
        }));
        assert!(holder.is_err(), "the holder was supposed to panic");

        drop(pool.get_with(|| counting_init(&built)));
        assert_eq!(
            built.load(Ordering::Relaxed),
            1,
            "the panicking holder's scratch never made it back to the pool"
        );
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
        assert_eq!(pipeline.inner.scratch_pool.len(), 1);
        let scratch = pipeline.inner.scratch_pool.get(&pipeline.inner.model);
        assert!(
            matches!(scratch.model, PipelineModelScratch::BPE(_)),
            "the pooled scratch is not the BPE scratch the model used"
        );
    }

    /// Where the buffers of the pooled scratch live, and how much room each has. The capacities
    /// are what keep the addresses honest: a `Vec` that never allocated owns no buffer, and every
    /// one of those reports the same dangling address.
    #[derive(Debug, PartialEq)]
    struct Buffers {
        pre_tokens: (usize, usize),
        symbols: (usize, usize),
    }

    fn pooled_buffers(pipeline: &PipelineTokenizer) -> Buffers {
        let scratch = pipeline.inner.scratch_pool.get(&pipeline.inner.model);
        let PipelineModelScratch::BPE(bpe) = &scratch.model else {
            panic!("a BPE pipeline encodes with a BPE scratch");
        };
        Buffers {
            pre_tokens: (
                scratch.pre_tokens.as_ptr() as usize,
                scratch.pre_tokens.capacity(),
            ),
            symbols: (bpe.symbols.as_ptr() as usize, bpe.symbols.capacity()),
        }
    }

    // What pooling buys is the allocation the previous call already made, for the pre-token spans
    // as much as for the model's own buffers. A warm scratch must therefore come back with the
    // same buffers, not merely with buffers large enough: an address moves when a `Vec`
    // reallocates, so this catches a call that replaces or regrows what the last one left.
    //
    // "helo " and not "hello ": a word that is itself a vocabulary entry is answered by the fold,
    // and the merge engines that write the symbols never run.
    #[test]
    fn a_warm_scratch_reuses_the_same_allocations() {
        let pipeline = hello_pipeline_split_on_words();
        let words = 50;
        let input = "helo ".repeat(words);

        // The first encode grows the buffers to what this input needs, the second runs inside
        // them. Anything from there on has nothing left to grow.
        pipeline.encode(input.as_str(), false).wait().unwrap();
        pipeline.encode(input.as_str(), false).wait().unwrap();
        let warm = pooled_buffers(&pipeline);
        assert!(
            warm.pre_tokens.1 >= words && warm.symbols.1 > 0,
            "the encodes left buffers this test cannot compare: {warm:?}"
        );

        pipeline.encode(input.as_str(), false).wait().unwrap();
        assert_eq!(
            pooled_buffers(&pipeline),
            warm,
            "a warm encode did not write into the buffers the pool handed it"
        );
    }

    /// The addresses of the three span buffers a sequence encode rotates between.
    fn pooled_span_buffers(pipeline: &PipelineTokenizer) -> BTreeSet<usize> {
        let scratch = pipeline.inner.scratch_pool.get(&pipeline.inner.model);
        let [first, second] = &scratch.split.pair;
        [&scratch.pre_tokens, first, second]
            .map(|spans| spans.as_ptr() as usize)
            .into_iter()
            .collect()
    }

    // `PipelineSequence` swaps its two buffers with `pre_tokens` rather than copying out of them,
    // so no one buffer keeps the same allocation from call to call and the per-buffer comparison
    // above would not hold here. What must hold is that the three allocations a warm pipeline
    // rotates are still the three it uses: a call that allocated a fresh buffer instead of taking
    // the pooled one would show an address the warm set does not have.
    #[test]
    fn a_warm_sequence_reuses_the_same_span_buffers() {
        let pipeline = hello_pipeline_sequence();
        let input = "helo ".repeat(50);

        // The buffers rotate, so they take a few encodes to all have been grown once.
        for _ in 0..4 {
            pipeline.encode(input.as_str(), false).wait().unwrap();
        }
        let warm = pooled_span_buffers(&pipeline);
        assert_eq!(
            warm.len(),
            3,
            "the encodes left buffers this test cannot compare: a `Vec` that never allocated \
             reports the same dangling address as every other one, so a set smaller than three \
             is comparing addresses that prove nothing"
        );

        pipeline.encode(input.as_str(), false).wait().unwrap();
        assert_eq!(
            pooled_span_buffers(&pipeline),
            warm,
            "a warm sequence encode allocated a span buffer instead of reusing the pooled ones"
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

        let mut scratch = pipeline.inner.scratch_pool.get(&pipeline.inner.model);
        let PipelineModelScratch::BPE(bpe) = &mut scratch.model else {
            panic!("a BPE pipeline encodes with a BPE scratch");
        };
        let cache = bpe.word_cache.as_mut().expect("BPE encodes with a cache");
        assert_eq!(cache.lookup(b"helo").hit(), Some(&[5u32, 3][..]));
    }
}
