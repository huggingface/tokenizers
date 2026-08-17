use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::parallelism::pool;
use rayon::prelude::*;

use crate::pipeline::{
    BatchEncoding, EncodeHandle, Encoding, Input, Inputs, PipelineToken, PipelineTokenizer,
    Segment, Seq, SpecialSegmentIterator,
};

use super::Result;

/// Threshold below which [`Inputs`] are encoded serially on the caller thread
pub(crate) const PARALLEL_MIN_BYTES: usize = 8 * 1024;

/// Upper bound on units claimed per cursor RMW; see [`claim_size`].
const MAX_CLAIM: usize = 64;

/// How many units one worker claims per cursor RMW.
///
/// Sized so a claim carries roughly [`PARALLEL_MIN_BYTES`] of text: with 8 KiB
/// documents that is one unit and scheduling is unchanged, while with 200-byte
/// chat lines it is tens of them.
///
/// The point is not really the saved atomics — claiming in runs was measured
/// worthless on its own. It is that a worker which claims a run of short
/// documents now *owns* those documents end to end, reconstructing each one
/// itself, so the ids are allocated and finished on the same thread instead of
/// being handed to the consumer to finish and free.
///
/// Capped so there are still several claims per thread: a claim is the
/// granularity at which work balances.
fn claim_size(units: &[Unit], threads: usize) -> usize {
    debug_assert!(!units.is_empty(), "callers return early on an empty plan");
    let total: usize = units.iter().map(|u| u.range.len()).sum();
    let avg = (total / units.len()).max(1);
    let by_bytes = PARALLEL_MIN_BYTES.div_ceil(avg);
    let by_balance = units.len() / threads.max(1) / 4;
    by_bytes.min(by_balance).clamp(1, MAX_CLAIM)
}

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

struct UnitResult(UnsafeCell<Option<Result<Vec<PipelineToken>>>>);

/// SAFETY: we make sure each unit's result is set by only one given thread at a time thanks to
/// [`Job::next_unit`]
unsafe impl Sync for UnitResult {}

impl UnitResult {
    fn new(tokens: Option<Result<Vec<PipelineToken>>>) -> Self {
        Self(UnsafeCell::new(tokens))
    }

    unsafe fn set(&self, tokens: Result<Vec<PipelineToken>>) {
        unsafe { *self.0.get() = Some(tokens) };
    }

    // TODO: safety doc
    fn take(&self) -> Option<Result<Vec<PipelineToken>>> {
        unsafe { &mut *self.0.get() }.take()
    }
}

struct Unit {
    /// Sequence index in the [`Inputs`] batch
    seq: usize,
    /// The unit's index within a sequence, because a given sequence can be split into multiple units for better parallelism
    idx: usize,
    /// Which member of the input pair ([`Seq::A`] for [`Input::Single`])
    side: Seq,
    /// The range within the input sequence to encode
    range: Range<usize>,
}

struct Plan {
    /// Units of work created based on the inputs
    units: Vec<Unit>,
    /// Contains the per-sequence length of [`Seq::A`] of a pair of inputs ([`Input::Single`] is
    /// always considered [`Seq::A`])
    side_a_len: Vec<usize>,
    /// Empty pre-allocated output buffer
    outputs: Vec<Vec<UnitResult>>,
}

/// A finished sequence, published by whichever worker completed its last unit.
struct EncodingSlot(UnsafeCell<Option<Result<Encoding>>>);

/// SAFETY: exactly one worker writes a given slot — the one whose `remaining`
/// decrement returned 1 — and it does so before publishing the sequence to the
/// completion queue with a Release store. The consumer only reads a slot after
/// seeing that publication.
unsafe impl Sync for EncodingSlot {}

impl EncodingSlot {
    fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    unsafe fn set(&self, enc: Result<Encoding>) {
        unsafe { *self.0.get() = Some(enc) };
    }

    fn take(&self) -> Option<Result<Encoding>> {
        unsafe { &mut *self.0.get() }.take()
    }
}

struct Job {
    inputs: Box<Inputs>,
    units: Vec<Unit>,
    outputs: Vec<Vec<UnitResult>>,
    cancelled: AtomicBool,
    /// Cursor for threads to pick up the next [`Unit`] to work on
    next_unit: CachePadded<AtomicUsize>,
    tokenizer: PipelineTokenizer,
    add_special_tokens: bool,
    /// Completion queue of *sequences* that are finished and reconstructed, in
    /// completion order. One entry per sequence: reconstruction now happens on the
    /// worker that finishes a sequence, so the consumer no longer counts units.
    completion_queue: Vec<AtomicUsize>,
    /// Cursor that each thread increments to publish a finished sequence
    next_completed: CachePadded<AtomicUsize>,
    /// Units still outstanding per sequence. The worker whose decrement returns 1
    /// owns reconstruction of that sequence.
    remaining: Vec<AtomicUsize>,
    /// Reconstructed sequences, written by the owning worker
    finished: Vec<EncodingSlot>,
    side_a_len: Vec<usize>,
    /// How many units a worker takes per claim; see [`claim_size`]
    claim: usize,
}

impl Job {
    const NOT_DONE: usize = usize::MAX;

    fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Sequence index published at completion slot `i`, or `None` if no unit has finished
    fn completed_seq(&self, i: usize) -> Option<usize> {
        match self.completion_queue[i].load(Ordering::Acquire) {
            Self::NOT_DONE => None,
            seq => Some(seq),
        }
    }

    /// Claim the next run of units, encode them, and reconstruct every sequence
    /// this worker finishes. Returns false once the cursor is past the end.
    ///
    /// Reconstruction used to happen on the consumer thread, one document at a
    /// time, which made it a serial stage that every document passed through. That
    /// is invisible on 8 KiB documents and decisive on 200-byte ones — it is why
    /// batches of short documents peaked at 2-4 threads and then got *slower*.
    /// Doing it here also means the ids are allocated and freed on the same thread,
    /// instead of the worker allocating and the consumer freeing.
    fn encode_claim(&self) -> bool {
        if self.cancelled.load(Ordering::Relaxed) {
            return false;
        }
        let start = self.next_unit.0.fetch_add(self.claim, Ordering::Relaxed);
        if start >= self.units.len() {
            return false;
        }
        let end = (start + self.claim).min(self.units.len());

        // One scratch buffer for the whole claim. Borrowing it costs a lock on a
        // process-wide mutex, and `encode_sequence` used to take that lock per
        // input: fine at 8 KiB a document, the entire bottleneck at 200 bytes.
        let mut scratch = self.tokenizer.scratch();

        for unit in &self.units[start..end] {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let input = self
                    .inputs
                    .get(unit.seq)
                    .ok_or_else(|| format!("invalid unit index: {}", unit.seq))?;
                let input = match input {
                    Input::Single(s) => s,
                    Input::Pair(s1, s2) => match unit.side {
                        Seq::A => s1,
                        Seq::B => s2,
                    },
                };
                let text = &input[unit.range.clone()];
                let mut out = Vec::with_capacity(text.len() / 4);
                self.tokenizer
                    .encode_sequence_scratched(text, &mut scratch, &mut out)?;
                Ok(out)
            }))
            .unwrap_or_else(|_| Err("encode worker panicked".into()));
            // SAFETY: no two threads can share the same unit of work because the
            // atomic fetch_add above hands each claimed range to exactly one thread
            unsafe { self.outputs[unit.seq][unit.idx].set(res) };

            // AcqRel: the release half publishes this unit's write to whichever
            // thread takes the sequence; the acquire half means the thread that
            // does take it (the one reading 1) sees every other unit's write.
            if self.remaining[unit.seq].fetch_sub(1, Ordering::AcqRel) == 1 {
                let enc = self.reconstruct(unit.seq);
                // SAFETY: exactly one thread observes the 1 -> 0 transition, so
                // exactly one thread writes this slot, and it does so before the
                // Release store below that makes the sequence visible.
                unsafe { self.finished[unit.seq].set(enc) };
                let slot = self.next_completed.0.fetch_add(1, Ordering::Relaxed);
                self.completion_queue[slot].store(unit.seq, Ordering::Release);
            }
        }

        true
    }

    /// Stitch a sequence's unit results together and post-process them. Runs on
    /// the worker that finished the sequence.
    fn reconstruct(&self, seq: usize) -> Result<Encoding> {
        let unit_results = &self.outputs[seq];
        let a_len = self.side_a_len[seq];

        let a = drain(&unit_results[..a_len])?;
        let b = (a_len < unit_results.len())
            .then(|| drain(&unit_results[a_len..]))
            .transpose()?;

        Ok(self.tokenizer.post_process(a, b, self.add_special_tokens))
    }

    fn take_finished(&self, seq: usize) -> Result<Encoding> {
        self.finished[seq]
            .take()
            .expect("[BUG] sequence published to the completion queue but not reconstructed")
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

pub(crate) struct StreamingIter {
    job: Arc<Job>,
    /// Next completion-queue slot to read. One slot per sequence now, so the
    /// per-sequence unit tally the consumer used to keep is gone.
    next_slot: usize,
    completed: usize,
}

impl StreamingIter {
    fn new(job: Arc<Job>) -> Self {
        Self {
            job,
            next_slot: 0,
            completed: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.job.len()
    }
}

impl Iterator for StreamingIter {
    type Item = (usize, Result<Encoding>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed == self.job.len() {
            return None;
        }
        loop {
            let Some(seq) = self.job.completed_seq(self.next_slot) else {
                // XXX: the caller has to wait for results regardless: instead of parking or empty
                // spinning, we do useful work
                if !self.job.encode_claim() {
                    std::hint::spin_loop();
                }
                continue;
            };
            self.next_slot += 1;
            self.completed += 1;
            return Some((seq, self.job.take_finished(seq)));
        }
    }
}

impl Drop for StreamingIter {
    fn drop(&mut self) {
        self.job.cancel();
    }
}

fn drain(results: &[UnitResult]) -> Result<Vec<PipelineToken>> {
    if let [only] = results {
        return only
            .take()
            .expect("[BUG] failed to take the unit's result when we expect it to be present");
    }
    let mut out = Vec::with_capacity(results.len());
    for res in results {
        match res.take() {
            Some(Ok(tokens)) => out.push(tokens),
            Some(Err(e)) => return Err(e),
            None => {
                unreachable!(
                    "[BUG] failed to take the unit's result when we expect it to be present"
                )
            }
        }
    }
    Ok(out.concat())
}

impl PipelineTokenizer {
    fn plan_sequence(
        &self,
        seq_idx: usize,
        side: Seq,
        input: &str,
        units: &mut Vec<Unit>,
        seq_outputs: &mut Vec<UnitResult>,
    ) {
        // If input is not at least twice the size of the minimum meaningful parallel
        // chunk's size, we emit the full input as its own chunk because splitting would be inefficient
        if input.len() < 2 * PARALLEL_MIN_BYTES {
            units.push(Unit {
                seq: seq_idx,
                idx: seq_outputs.len(),
                side,
                range: 0..input.len(),
            });
            seq_outputs.push(UnitResult::new(None));
            return;
        }
        let current_units_len = units.len();
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            let idx = seq_outputs.len();
            let (segment, offset) = match segment {
                Segment::SpecialToken(id) => {
                    let token = PipelineToken::from(id);
                    seq_outputs.push(UnitResult::new(Some(Ok(vec![token]))));
                    continue;
                }
                Segment::Text { text, input_offset } => (text, input_offset),
            };
            units.push(Unit {
                seq: seq_idx,
                idx,
                side,
                range: offset..offset + segment.len(),
            });
            seq_outputs.push(UnitResult::new(None));
        }
        // make sure we have at least one unit per sequence, otherwise we'll wait indefinitely
        // for a completion event
        if current_units_len == units.len() {
            let idx = seq_outputs.len();
            // sentinel unit resulting in an encode no-op since range is 0..0
            units.push(Unit {
                seq: seq_idx,
                idx,
                side,
                range: 0..0,
            });
            seq_outputs.push(UnitResult::new(None));
        }
    }

    fn plan_work(&self, inputs: &Inputs) -> Plan {
        let mut units = Vec::with_capacity(inputs.len());
        let mut side_a_len = Vec::with_capacity(inputs.len());
        let mut outputs = Vec::with_capacity(inputs.len());
        for (seq_idx, input) in inputs.into_iter().enumerate() {
            let mut seq_outputs = vec![];
            match input {
                Input::Single(s) => {
                    self.plan_sequence(seq_idx, Seq::A, s, &mut units, &mut seq_outputs);
                    side_a_len.push(seq_outputs.len());
                }
                Input::Pair(s1, s2) => {
                    self.plan_sequence(seq_idx, Seq::A, s1, &mut units, &mut seq_outputs);
                    let a_len = seq_outputs.len();
                    self.plan_sequence(seq_idx, Seq::B, s2, &mut units, &mut seq_outputs);
                    side_a_len.push(a_len);
                }
            }
            outputs.push(seq_outputs);
        }
        // Schedule encode batch by longest unit first
        units.sort_unstable_by_key(|u| std::cmp::Reverse(u.range.len()));
        Plan {
            units,
            side_a_len,
            outputs,
        }
    }
}

/// Flat batch encode across the pool: each chunk of documents fills its own arena,
/// then the arenas are concatenated once.
///
/// Allocations are per *chunk*, not per document — a 26k-document batch costs a
/// couple of hundred instead of 26k, and none of them is freed by a different
/// thread than allocated it. Returns `Ok(None)` when no pool is available so the
/// caller can run its serial path.
pub(crate) fn encode_flat(
    tok: &PipelineTokenizer,
    inputs: &[&str],
    prefix: &[PipelineToken],
    suffix: &[PipelineToken],
) -> Result<Option<BatchEncoding>> {
    let Some(pool) = pool() else {
        return Ok(None);
    };
    let threads = pool.current_num_threads();
    if threads < 2 {
        return Ok(None);
    }

    // Chunks sized so each carries real work but every thread still gets several.
    let total: usize = inputs.iter().map(|s| s.len()).sum();
    let avg = (total / inputs.len().max(1)).max(1);
    let by_bytes = PARALLEL_MIN_BYTES.div_ceil(avg);
    let by_balance = (inputs.len() / (threads * 4).max(1)).max(1);
    let chunk = by_bytes.min(by_balance).max(1);

    let parts: Vec<Result<(Vec<PipelineToken>, Vec<u32>)>> = pool.install(|| {
        inputs
            .par_chunks(chunk)
            .map(|docs| {
                let bytes: usize = docs.iter().map(|s| s.len()).sum();
                let mut arena =
                    Vec::with_capacity(bytes / 4 + (prefix.len() + suffix.len()) * docs.len());
                let mut lens = Vec::with_capacity(docs.len());
                // One scratch for the whole chunk: see `PipelineTokenizer::scratch`.
                let mut scratch = tok.scratch();
                for doc in docs {
                    let start = arena.len();
                    arena.extend_from_slice(prefix);
                    tok.encode_sequence_scratched(doc, &mut scratch, &mut arena)?;
                    arena.extend_from_slice(suffix);
                    lens.push((arena.len() - start) as u32);
                }
                Ok((arena, lens))
            })
            .collect()
    });

    let parts = parts.into_iter().collect::<Result<Vec<_>>>()?;

    // Concatenate the arenas in input order, then walk the recorded per-document
    // lengths to turn them into row starts.
    let total_ids: usize = parts.iter().map(|(arena, _)| arena.len()).sum();
    let mut ids = Vec::with_capacity(total_ids);
    let mut offsets = Vec::with_capacity(inputs.len() + 1);
    let mut at = 0u32;
    for (arena, lens) in &parts {
        ids.extend_from_slice(arena);
        for len in lens {
            offsets.push(at);
            at += *len;
        }
    }
    offsets.push(at);
    debug_assert_eq!(offsets.len(), inputs.len() + 1);
    debug_assert_eq!(at as usize, ids.len());

    Ok(Some(BatchEncoding::from_parts(ids, offsets)))
}

pub(crate) fn encode(
    tok: &PipelineTokenizer,
    inputs: Inputs,
    add_special_tokens: bool,
) -> EncodeHandle {
    if inputs.size_bytes() < PARALLEL_MIN_BYTES {
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    }
    let Some(pool) = pool() else {
        // unable to get a pool handle, reverting to single threaded
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    };
    let Plan {
        units,
        side_a_len,
        outputs,
    } = tok.plan_work(&inputs);
    assert!(
        units.len() < usize::MAX,
        "Job::next_unit cursor will overflow messing up internals if we have more units of work than usize::MAX"
    );
    if units.len() < 2 {
        return EncodeHandle::blocking(tok.encode_serial(inputs, add_special_tokens));
    }
    let threads = units.len().min(pool.current_num_threads());
    let job = Arc::new(Job {
        inputs: Box::new(inputs),
        cancelled: AtomicBool::new(false),
        next_unit: CachePadded(AtomicUsize::new(0)),
        add_special_tokens,
        tokenizer: tok.clone(),
        // One slot per sequence, not per unit: workers publish finished sequences.
        completion_queue: (0..outputs.len())
            .map(|_| AtomicUsize::new(Job::NOT_DONE))
            .collect(),
        next_completed: CachePadded(AtomicUsize::new(0)),
        remaining: units
            .iter()
            .fold(vec![0usize; outputs.len()], |mut uc, u| {
                uc[u.seq] += 1;
                uc
            })
            .into_iter()
            .map(AtomicUsize::new)
            .collect(),
        finished: (0..outputs.len()).map(|_| EncodingSlot::new()).collect(),
        claim: claim_size(&units, threads),
        outputs,
        side_a_len,
        units,
    });
    for _ in 0..threads {
        let job = job.clone();
        pool.spawn(move || while job.encode_claim() {});
    }
    EncodeHandle::streaming(StreamingIter::new(job))
}
