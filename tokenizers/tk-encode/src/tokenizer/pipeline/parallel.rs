use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::parallelism::pool;
use crate::pipeline::scratch_pool::{EncodeScratch, ScratchGuard};
use crate::pipeline::{
    EncodeHandle, Encoding, Input, Inputs, PipelineToken, PipelineTokenizer, Segment, Seq,
    SpecialSegmentIterator,
};

use super::Result;

/// Threshold below which [`Inputs`] are encoded serially on the caller thread
// `pub` (re-exported by `pipeline`): the differential "parallel == serial" tests live in
// `tk-convert` now — they need a `Tokenizer` to build from — and have to size an input past
// this threshold to reach the parallel path at all.
pub const PARALLEL_MIN_BYTES: usize = 8 * 1024;

impl Input {
    fn len(&self) -> usize {
        match self {
            Self::Single(s) => s.len(),
            Self::Pair(s1, s2) => s1.len() + s2.len(),
        }
    }
}

impl Inputs {
    fn size_bytes(&self) -> usize {
        self.as_slice().iter().map(Input::len).sum()
    }

    fn get(&self, i: usize) -> Option<&Input> {
        self.as_slice().get(i)
    }
}

/// TODO: benchmark with and without to validate usefulness
/// Isolate an atomic in its own cache line to avoid excessive invalidation when multiple threads
/// access the value
#[repr(align(64))]
struct CachePadded<T>(pub(crate) T);

struct Slot<T>(UnsafeCell<Option<T>>);

impl<T> Slot<T> {
    fn new(value: Option<T>) -> Self {
        Self(UnsafeCell::new(value))
    }

    unsafe fn set(&self, value: T) {
        unsafe { *self.0.get() = Some(value) };
    }

    // TODO: safety doc
    fn take(&self) -> Option<T> {
        unsafe { &mut *self.0.get() }.take()
    }
}

type ChunkResult = Slot<Result<Vec<PipelineToken>>>;
/// SAFETY: we make sure each chunk's result is set by only one given thread thanks to
/// [`EncodeBatch::next_task`]
unsafe impl Sync for ChunkResult {}

type EncodingResult = Slot<Result<Encoding>>;
/// SAFETY: we make sure each result is set only by one thread thanks to [`EncodeBatch::remaining`]
unsafe impl Sync for EncodingResult {}

struct SequenceChunk {
    /// Sequence index in the [`Inputs`] batch
    seq: usize,
    /// The chunk's index within a sequence, because a given sequence can be split into multiple chunks for better parallelism
    idx: usize,
    /// Which member of the input pair ([`Seq::A`] for [`Input::Single`])
    side: Seq,
    /// The range within the input sequence to encode
    range: Range<usize>,
}

/// One bucket per power-of-two length, plus one for empty chunks.
const SIZE_CLASSES: usize = usize::BITS as usize + 1;

/// Which bucket a chunk of `len` bytes goes in. Class `k + 1` holds `2^k .. 2^(k+1)`, so walking
/// the classes downwards visits the chunks longest-group-first, which is all the task grouping
/// needs -- it accumulates bytes until it has a task's worth, and never asks for a total order.
#[inline]
fn size_class(len: usize) -> usize {
    if len == 0 {
        0
    } else {
        len.ilog2() as usize + 1
    }
}

/// Indices of `chunks` grouped by descending size class, and the per-sequence chunk counts, in one
/// pass each. A counting sort: histogram, exclusive prefix sums largest-class-first, scatter.
///
/// Moves 4-byte indices rather than 40-byte [`SequenceChunk`]s, and is O(n) rather than
/// O(n log n) -- which matters because this runs on the caller before any worker is spawned.
fn order_and_counts(chunks: &[SequenceChunk], n_seq: usize) -> (Vec<u32>, Vec<usize>) {
    let mut class_count = [0usize; SIZE_CLASSES];
    let mut chunk_count = vec![0usize; n_seq];
    for chunk in chunks {
        class_count[size_class(chunk.range.len())] += 1;
        chunk_count[chunk.seq] += 1;
    }
    let mut cursor = [0usize; SIZE_CLASSES];
    let mut at = 0;
    for class in (0..SIZE_CLASSES).rev() {
        cursor[class] = at;
        at += class_count[class];
    }
    let mut order = vec![0u32; chunks.len()];
    for (i, chunk) in chunks.iter().enumerate() {
        let slot = &mut cursor[size_class(chunk.range.len())];
        order[*slot] = i as u32;
        *slot += 1;
    }
    (order, chunk_count)
}

struct Plan {
    /// Sequence chunks created based on the inputs, in the order `plan_sequence` emitted them
    chunks: Vec<SequenceChunk>,
    /// Indices into [`Self::chunks`], grouped longest-class-first. [`Self::tasks`] are ranges over
    /// *this*, not over `chunks`.
    order: Vec<u32>,
    /// Groups of sequence chunks to be picked up by workers: this is used so that a given worker can process
    /// a given number of bytes in one go rather than contend on [`EncodeBatch`] (useful when lots of tiny chunks)
    tasks: Vec<Range<usize>>,
    /// Contains the per-sequence length of [`Seq::A`] of a pair of inputs ([`Input::Single`] is
    /// always considered [`Seq::A`])
    side_a_len: Vec<usize>,
    /// Empty pre-allocated output buffer
    outputs: Vec<Vec<ChunkResult>>,
    /// Number of chunks per sequence (accessed via `chunk_count[seq]`)
    chunk_count: Vec<usize>,
}

struct EncodeBatch {
    inputs: Box<Inputs>,
    chunks: Vec<SequenceChunk>,
    /// Indices into [`Self::chunks`], longest-class-first; [`Self::tasks`] index this.
    order: Vec<u32>,
    /// All the tasks to be picked up by workers: each tasks represents one or many [`SequenceChunk`]s to process
    tasks: Vec<Range<usize>>,
    outputs: Vec<Vec<ChunkResult>>,
    encodings: Vec<EncodingResult>,
    /// Chunks left to encode per sequence (remaining[seq] == chunks_left)
    remaining: Vec<AtomicUsize>,
    cancelled: AtomicBool,
    /// Cursor for threads to pick up the next task (group of sequence chunks) to work on
    next_task: CachePadded<AtomicUsize>,
    tokenizer: PipelineTokenizer,
    add_special_tokens: bool,
    /// Completion queue containing the list of chunks (inserted as their sequence idx) that have finished, in completion order
    /// This enables us to track the number of remaining chunks, to know when we can return a result
    completion_queue: Vec<AtomicUsize>,
    /// Cursor that each thread increments to add a completed chunk to the completion queue
    next_completed: CachePadded<AtomicUsize>,
    side_a_len: Vec<usize>,
}

impl EncodeBatch {
    const NOT_DONE: usize = usize::MAX;

    fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Sequence index published at completion slot `i`, or `None` if no chunk has finished
    fn completed_seq(&self, i: usize) -> Option<usize> {
        match self.completion_queue[i].load(Ordering::Acquire) {
            Self::NOT_DONE => None,
            seq => Some(seq),
        }
    }

    fn encode_task(&self, scratch: &mut EncodeScratch) -> bool {
        if self.cancelled.load(Ordering::Relaxed)
            || self.next_task.0.load(Ordering::Relaxed) >= self.tasks.len()
        {
            return false;
        }
        let t = self.next_task.0.fetch_add(1, Ordering::Relaxed);
        let Some(task) = self.tasks.get(t) else {
            return false;
        };

        let mut finished = Vec::new();
        for i in task.clone() {
            let chunk = &self.chunks[self.order[i] as usize];
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let input = self
                    .inputs
                    .get(chunk.seq)
                    .ok_or_else(|| format!("invalid chunk index: {}", chunk.seq))?;
                let input = match input {
                    Input::Single(s) => s,
                    Input::Pair(s1, s2) => match chunk.side {
                        Seq::A => s1,
                        Seq::B => s2,
                    },
                };
                self.tokenizer
                    .encode_sequence_with(&input[chunk.range.clone()], scratch)
            }))
            .unwrap_or_else(|_| Err("encode worker panicked".into()));
            // SAFETY: no two threads can share the same chunk because each chunk is owned by
            // only one task
            unsafe { self.outputs[chunk.seq][chunk.idx].set(res) };

            if self.remaining[chunk.seq].fetch_sub(1, Ordering::AcqRel) == 1 {
                let encoding = self.reconstruct(chunk.seq);
                // SAFETY: only the thread that decremented self.remaining[chunk.seq] to 0 writes to
                // the slot
                unsafe { self.encodings[chunk.seq].set(encoding) };
                finished.push(chunk.seq);
            }
        }

        if !finished.is_empty() {
            let base = self
                .next_completed
                .0
                .fetch_add(finished.len(), Ordering::Relaxed);
            for (i, seq) in finished.into_iter().enumerate() {
                self.completion_queue[base + i].store(seq, Ordering::Release);
            }
        }

        true
    }

    fn reconstruct(&self, seq: usize) -> Result<Encoding> {
        let chunk_results = &self.outputs[seq];
        let a_len = self.side_a_len[seq];
        let a = drain(&chunk_results[..a_len])?;
        let b = (a_len < chunk_results.len())
            .then(|| drain(&chunk_results[a_len..]))
            .transpose()?;
        self.tokenizer.post_process(a, b, self.add_special_tokens)
    }

    fn take_encoding(&self, seq: usize) -> Result<Encoding> {
        self.encodings[seq]
            .take()
            .expect("[BUG] completion signalled before the encoding was published")
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

pub(crate) struct StreamingIter {
    batch: Arc<EncodeBatch>,
    next: usize,
    completed: usize,
}

impl StreamingIter {
    fn new(batch: Arc<EncodeBatch>) -> Self {
        Self {
            batch,
            next: 0,
            completed: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.batch.len()
    }
}

impl Iterator for StreamingIter {
    type Item = (usize, Result<Encoding>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed == self.batch.len() {
            return None;
        }
        let mut scratch = None;
        loop {
            if let Some(seq) = self.batch.completed_seq(self.next) {
                self.next += 1;
                self.completed += 1;
                return Some((seq, self.batch.take_encoding(seq)));
            }
            let scratch = scratch.get_or_insert_with(|| self.batch.tokenizer.get_scratch());
            // The caller has to wait for results regardless: instead of parking or empty
            // spinning, we do useful work
            if !self.batch.encode_task(scratch) {
                std::hint::spin_loop();
            }
        }
    }
}

impl Drop for StreamingIter {
    fn drop(&mut self) {
        self.batch.cancel();
    }
}

fn drain(results: &[ChunkResult]) -> Result<Vec<PipelineToken>> {
    if let [only] = results {
        return only
            .take()
            .expect("[BUG] failed to take the chunk's result when we expect it to be present");
    }
    let mut out = Vec::with_capacity(results.len());
    for res in results {
        match res.take() {
            Some(Ok(tokens)) => out.push(tokens),
            Some(Err(e)) => return Err(e),
            None => {
                unreachable!(
                    "[BUG] failed to take the chunk's result when we expect it to be present"
                )
            }
        }
    }
    Ok(out.concat())
}

impl PipelineTokenizer {
    fn get_scratch(&self) -> ScratchGuard<'_> {
        self.inner.scratch_pool.get(&self.inner.model)
    }

    fn plan_sequence(
        &self,
        seq_idx: usize,
        side: Seq,
        input: &str,
        chunks: &mut Vec<SequenceChunk>,
        seq_outputs: &mut Vec<ChunkResult>,
    ) {
        // If input is not at least twice the size of the minimum meaningful parallel
        // chunk's size, we emit the full input as its own chunk because splitting would be inefficient
        if input.len() < 2 * PARALLEL_MIN_BYTES {
            chunks.push(SequenceChunk {
                seq: seq_idx,
                idx: seq_outputs.len(),
                side,
                range: 0..input.len(),
            });
            seq_outputs.push(ChunkResult::new(None));
            return;
        }
        let current_chunks_len = chunks.len();
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            let idx = seq_outputs.len();
            let (segment, offset) = match segment {
                Segment::SpecialToken(id) => {
                    let token = PipelineToken::from(id);
                    seq_outputs.push(ChunkResult::new(Some(Ok(vec![token]))));
                    continue;
                }
                Segment::Text { text, input_offset } => (text, input_offset),
            };
            chunks.push(SequenceChunk {
                seq: seq_idx,
                idx,
                side,
                range: offset..offset + segment.len(),
            });
            seq_outputs.push(ChunkResult::new(None));
        }
        // make sure we have at least one chunk per sequence, otherwise we'll wait indefinitely
        // for a completion event
        if current_chunks_len == chunks.len() {
            let idx = seq_outputs.len();
            // sentinel chunk resulting in an encode no-op since range is 0..0
            chunks.push(SequenceChunk {
                seq: seq_idx,
                idx,
                side,
                range: 0..0,
            });
            seq_outputs.push(ChunkResult::new(None));
        }
    }

    fn plan_work(&self, inputs: &Inputs) -> Plan {
        let mut chunks = Vec::with_capacity(inputs.len());
        let mut side_a_len = Vec::with_capacity(inputs.len());
        let mut outputs = Vec::with_capacity(inputs.len());
        for (seq_idx, input) in inputs.into_iter().enumerate() {
            let mut seq_outputs = vec![];
            match input {
                Input::Single(s) => {
                    self.plan_sequence(seq_idx, Seq::A, s, &mut chunks, &mut seq_outputs);
                    side_a_len.push(seq_outputs.len());
                }
                Input::Pair(s1, s2) => {
                    self.plan_sequence(seq_idx, Seq::A, s1, &mut chunks, &mut seq_outputs);
                    let a_len = seq_outputs.len();
                    self.plan_sequence(seq_idx, Seq::B, s2, &mut chunks, &mut seq_outputs);
                    side_a_len.push(a_len);
                }
            }
            outputs.push(seq_outputs);
        }
        // Schedule the batch longest-chunk-group first.
        //
        // This used to be `chunks.sort_unstable_by_key(Reverse(len))` plus a separate fold for
        // `chunk_count`. Two costs there: the sort is O(n log n) *moves of a 40-byte
        // `SequenceChunk`*, and it runs in the serial prologue -- on the caller, before a single
        // worker is spawned -- so it is pure Amdahl fraction, worst exactly in the many-small-inputs
        // case tasks were introduced for (one chunk per sequence means n == the batch size).
        //
        // The grouping below never needs a total order: it walks the chunks accumulating bytes and
        // cuts a task every `PARALLEL_MIN_BYTES`. Ordering by power-of-two size class is enough, and
        // that is a counting sort: one pass to histogram, one to scatter, moving 4-byte indices
        // instead of 40-byte structs. The per-sequence `chunk_count` comes off the same pass.
        let (order, chunk_count) = order_and_counts(&chunks, outputs.len());

        let mut tasks = Vec::new();
        let mut start = 0;
        let mut acc = 0;

        for (i, &c) in order.iter().enumerate() {
            acc += chunks[c as usize].range.len();
            if acc >= PARALLEL_MIN_BYTES {
                tasks.push(start..i + 1);
                start = i + 1;
                acc = 0;
            }
        }

        if start < order.len() {
            tasks.push(start..order.len());
        }

        Plan {
            chunks,
            order,
            tasks,
            side_a_len,
            outputs,
            chunk_count,
        }
    }
}

pub(crate) fn encode(
    tok: &PipelineTokenizer,
    inputs: Inputs,
    add_special_tokens: bool,
) -> EncodeHandle {
    if inputs.size_bytes() < PARALLEL_MIN_BYTES {
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    }
    // One sequence under `2 * PARALLEL_MIN_BYTES` is emitted by `plan_sequence` as a single chunk,
    // which becomes a single task, which fails the `tasks.len() < 2` check below -- but only after
    // `plan_work` has allocated `chunks`, `outputs`, `side_a_len`, `chunk_count` and `tasks`, and
    // thrown all five away. Every `encode` of a lone document between `PARALLEL_MIN_BYTES` and
    // twice it paid for that. Decide it before paying.
    //
    // `Input::Single` only: a pair is two `plan_sequence` calls, so it can reach two tasks and has
    // to go through planning to find out.
    if let Inputs::Single(Input::Single(seq)) = &inputs
        && seq.len() < 2 * PARALLEL_MIN_BYTES
    {
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    }
    let Some(pool) = pool() else {
        // unable to get a pool handle, reverting to single threaded
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    };
    let Plan {
        chunks,
        order,
        tasks,
        side_a_len,
        outputs,
        chunk_count,
    } = tok.plan_work(&inputs);
    if tasks.len() < 2 {
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    }
    let n_seq = outputs.len();
    let threads = tasks.len().min(pool.current_num_threads());
    let batch = Arc::new(EncodeBatch {
        inputs: Box::new(inputs),
        cancelled: AtomicBool::new(false),
        next_task: CachePadded(AtomicUsize::new(0)),
        add_special_tokens,
        tokenizer: tok.clone(),
        completion_queue: (0..n_seq)
            .map(|_| AtomicUsize::new(EncodeBatch::NOT_DONE))
            .collect(),
        next_completed: CachePadded(AtomicUsize::new(0)),
        remaining: chunk_count.iter().map(|&uc| AtomicUsize::new(uc)).collect(),
        encodings: (0..n_seq).map(|_| Slot::new(None)).collect(),
        outputs,
        side_a_len,
        chunks,
        order,
        tasks,
    });
    for _ in 0..threads {
        let batch = batch.clone();
        pool.spawn(move || {
            let mut scratch = batch.tokenizer.get_scratch();
            while batch.encode_task(&mut scratch) {}
        });
    }
    EncodeHandle::streaming(StreamingIter::new(batch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(seq: usize, len: usize) -> SequenceChunk {
        SequenceChunk {
            seq,
            idx: 0,
            side: Seq::A,
            range: 0..len,
        }
    }

    #[test]
    fn size_class_is_monotonic_in_length() {
        assert_eq!(size_class(0), 0);
        let mut last = 0;
        for len in [1usize, 2, 3, 4, 7, 8, 4095, 4096, 8191, 8192, 1 << 30] {
            let class = size_class(len);
            assert!(class >= last, "class fell from {last} at len {len}");
            last = class;
        }
        // A class holds exactly one power-of-two band.
        assert_eq!(size_class(8), size_class(15));
        assert!(size_class(16) > size_class(15));
    }

    #[test]
    fn the_order_is_a_permutation_grouped_largest_class_first() {
        let lens = [1usize, 70_000, 3, 8192, 0, 8191, 300, 70_001];
        let chunks: Vec<_> = lens.iter().map(|&l| chunk(0, l)).collect();
        let (order, _) = order_and_counts(&chunks, 1);

        // Every index exactly once.
        let mut seen: Vec<u32> = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..lens.len() as u32).collect::<Vec<_>>());

        // Classes never increase as we walk it, so the byte-accumulating task grouping meets the
        // big chunks first and leaves a short tail.
        let classes: Vec<usize> = order
            .iter()
            .map(|&i| size_class(chunks[i as usize].range.len()))
            .collect();
        assert!(
            classes.windows(2).all(|w| w[0] >= w[1]),
            "classes not descending: {classes:?}"
        );
        // The two 70k chunks lead and the empty one is last.
        assert_eq!(chunks[order[0] as usize].range.len() / 1000, 70);
        assert_eq!(chunks[*order.last().unwrap() as usize].range.len(), 0);
    }

    #[test]
    fn counts_are_per_sequence() {
        let chunks = vec![chunk(0, 10), chunk(2, 20), chunk(0, 30), chunk(2, 40)];
        let (order, counts) = order_and_counts(&chunks, 3);
        assert_eq!(counts, vec![2, 0, 2]);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn an_empty_plan_orders_nothing() {
        let (order, counts) = order_and_counts(&[], 2);
        assert!(order.is_empty());
        assert_eq!(counts, vec![0, 0]);
    }
}
