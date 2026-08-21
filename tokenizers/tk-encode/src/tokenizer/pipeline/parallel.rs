use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::parallelism::pool;
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

struct Job {
    inputs: Box<Inputs>,
    units: Vec<Unit>,
    outputs: Vec<Vec<UnitResult>>,
    cancelled: AtomicBool,
    /// Cursor for threads to pick up the next [`Unit`] to work on
    next_unit: CachePadded<AtomicUsize>,
    tokenizer: PipelineTokenizer,
    add_special_tokens: bool,
    /// Completion queue containing the list of units (inserted as their sequence idx) that have finished, in completion order
    /// this enables us to increment a counter consumer side so that when collected[seq] == unit_count[seq], we know we can return a result
    completion_queue: Vec<AtomicUsize>,
    /// Cursor that each thread increments to add a completed unit to the completion queue
    next_completed: CachePadded<AtomicUsize>,
    /// Number of units per sequence (accessed via `unit_count[seq]`)
    unit_count: Vec<usize>,
    side_a_len: Vec<usize>,
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

    /// Number of units that make up sequence `seq`
    fn unit_count(&self, seq: usize) -> usize {
        self.unit_count[seq]
    }

    fn encode_unit(&self) -> bool {
        if self.cancelled.load(Ordering::Relaxed) {
            return false;
        }
        let i = self.next_unit.0.fetch_add(1, Ordering::Relaxed);
        if i >= self.units.len() {
            return false;
        }

        let unit = &self.units[i];
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
            self.tokenizer.encode_sequence(&input[unit.range.clone()])
        }))
        .unwrap_or_else(|_| Err("encode worker panicked".into()));
        // SAFETY: no two threads can share the same unit of work because of the atomic fetch_add
        // call on the cursor
        unsafe { self.outputs[unit.seq][unit.idx].set(res) };

        let next = self.next_completed.0.fetch_add(1, Ordering::Relaxed);
        self.completion_queue[next].store(unit.seq, Ordering::Release);

        true
    }

    fn take_result(&self, seq: usize) -> Result<Encoding> {
        // NOTE: in a previous implementation, we implemented the reconstruction in parallel on the
        // thread pool; perhaps something to reconsider if the reconstruction shows up in profiling

        let unit_results = &self.outputs[seq];
        let a_len = self.side_a_len[seq];

        let a = drain(&unit_results[..a_len])?;
        let b = (a_len < unit_results.len())
            .then(|| drain(&unit_results[a_len..]))
            .transpose()?;

        Ok(self.tokenizer.post_process(a, b, self.add_special_tokens))
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

pub(crate) struct StreamingIter {
    job: Arc<Job>,
    next_unit: usize,
    collected: Vec<usize>,
    completed: usize,
}

impl StreamingIter {
    fn new(job: Arc<Job>) -> Self {
        let collected = vec![0; job.len()];
        Self {
            job,
            next_unit: 0,
            collected,
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
            let Some(seq) = self.job.completed_seq(self.next_unit) else {
                // XXX: the caller has to wait for results regardless: instead of parking or empty
                // spinning, we do useful work
                if !self.job.encode_unit() {
                    std::hint::spin_loop();
                }
                continue;
            };
            self.next_unit += 1;
            self.collected[seq] += 1;
            if self.collected[seq] == self.job.unit_count(seq) {
                self.completed += 1;
                return Some((seq, self.job.take_result(seq)));
            }
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
        completion_queue: (0..units.len())
            .map(|_| AtomicUsize::new(Job::NOT_DONE))
            .collect(),
        next_completed: CachePadded(AtomicUsize::new(0)),
        unit_count: units.iter().fold(vec![0; outputs.len()], |mut uc, u| {
            uc[u.seq] += 1;
            uc
        }),
        outputs,
        side_a_len,
        units,
    });
    for _ in 0..threads {
        let job = job.clone();
        pool.spawn(move || while job.encode_unit() {});
    }
    EncodeHandle::streaming(StreamingIter::new(job))
}
