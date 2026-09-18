//! The parallel batch encode: one contiguous id buffer, nothing shared between workers.

use crate::parallelism::pool;
use crate::pipeline::{Document, Encoding, PipelineToken, PipelineTokenizer, Template};
use crate::utils::padding::{PaddingDirection, PaddingParams, pad_length};

use super::Result;

/// What one worker hands back: its arena of ids, the length of each document in it, and the
/// longest of those, which is the stride when the batch is padded.
type Part = (Vec<PipelineToken>, Vec<u32>, u32);

/// Below this many bytes a batch is encoded serially: the pool costs more than it saves.
///
/// `pub` (re-exported by `pipeline`): the differential "parallel == serial" tests live in
/// `tk-convert`, and have to size an input past this threshold to reach the parallel path at all.
pub const PARALLEL_MIN_BYTES: usize = 8 * 1024;

/// The flat batch path: each worker takes a run of documents, encodes them into one arena of its
/// own, and records how long each came out. Nothing is shared, so there is no plan to build, no
/// completion queue to drain and no lock on the hot path -- the whole coordination structure the
/// general path needs exists to hand back a `Vec` per document, which this shape does not do.
///
/// Returns `None` when there is no pool or only one thread, so the caller runs its serial loop.
pub(crate) fn encode_flat(
    tok: &PipelineTokenizer,
    inputs: &[Document<'_>],
    template: &Template,
    add_special_tokens: bool,
    padding: Option<&PaddingParams>,
) -> Result<Option<Encoding>> {
    use rayon::prelude::*;

    let Some(pool) = pool() else {
        return Ok(None);
    };
    let threads = pool.current_num_threads();
    if threads < 2 {
        return Ok(None);
    }

    // Chunks sized so each carries real work but every thread still gets several of them.
    let total: usize = inputs.iter().map(|d| d.text.len()).sum();
    let avg = (total / inputs.len().max(1)).max(1);
    // Enough documents to be worth a task -- twice the parallel floor measured best, below which
    // rayon's per-task cost starts to show -- but never so many that a thread gets only one run.
    let by_bytes = (2 * PARALLEL_MIN_BYTES).div_ceil(avg);
    let by_balance = (inputs.len() / (threads * 4).max(1)).max(1);
    let chunk = by_bytes.min(by_balance).max(1);

    let laid_out = pool.install(|| -> Result<Encoding> {
        // Each worker fills an arena of its own and records how long every document came out.
        //
        // The arena is small enough to stay in cache, which is why the copy below is worth making:
        // writing straight into the shared output instead was measured slower -- every write then
        // lands somewhere in a multi-megabyte buffer and allocates a cache line from memory, while
        // this keeps the hot path local and pays one streaming copy at the end.
        let parts: Vec<Result<Part>> = inputs
            .par_chunks(chunk)
            // One scratch per worker, not one per chunk. `ScratchPool` is a mutex, and the scratch
            // it hands back carries a 2 MB word cache: taking one per chunk both contends on the
            // lock and drags that cache between cores as the pool reissues it.
            .map_init(
                || tok.scratch(),
                |scratch, docs| {
                    let bytes: usize = docs.iter().map(|d| d.text.len()).sum();
                    let mut arena =
                        Vec::with_capacity(bytes / 4 + template.n_special() * docs.len());
                    let mut lens = Vec::with_capacity(docs.len());
                    let mut longest = 0;
                    for doc in docs {
                        let start = arena.len();
                        tok.frame(
                            *doc,
                            template,
                            add_special_tokens,
                            scratch,
                            &mut arena,
                            None,
                        )?;
                        let len = (arena.len() - start) as u32;
                        longest = longest.max(len);
                        lens.push(len);
                    }
                    Ok((arena, lens, longest))
                },
            )
            .collect();
        let parts = parts.into_iter().collect::<Result<Vec<_>>>()?;

        let total_ids: usize = parts.iter().map(|(arena, ..)| arena.len()).sum();
        let total_rows: usize = parts.iter().map(|(_, lens, _)| lens.len()).sum();

        if let Some(params) = padding {
            // Each worker already tracked its own longest, so the stride costs one value per chunk
            // rather than a walk over every document. Serial work between two parallel phases is
            // what lets the pool's threads fall asleep, and waking them costs more than this did.
            let longest = parts.iter().map(|(.., l)| *l).max().unwrap_or(0) as usize;
            let stride = pad_length(longest, params).max(longest);
            let total = total_rows * stride;
            let pad = PipelineToken::from(params.pad_id);
            // Uninitialised: a worker owns its whole region, so it writes its documents *and*
            // their padding, and every slot is written exactly once.
            let mut ids: Vec<PipelineToken> = Vec::with_capacity(total);
            // The mask is the one buffer worth pre-filling: `vec![0u8; n]` is `alloc_zeroed`, free,
            // and the workers only mark the real tokens.
            let mut mask = vec![0u8; total];
            {
                let mut id_rest = &mut ids.spare_capacity_mut()[..total];
                let mut mask_rest = mask.as_mut_slice();
                let mut jobs = Vec::with_capacity(parts.len());
                for (arena, lens, _) in &parts {
                    let (id_dst, rest) = id_rest.split_at_mut(lens.len() * stride);
                    id_rest = rest;
                    let (mask_dst, rest) = mask_rest.split_at_mut(lens.len() * stride);
                    mask_rest = rest;
                    jobs.push((arena, lens, id_dst, mask_dst));
                }
                jobs.par_iter_mut()
                    .for_each(|(arena, lens, id_dst, mask_dst)| {
                        let mut read = 0;
                        for (document, &len) in lens.iter().enumerate() {
                            let len = len as usize;
                            let slot = &mut id_dst[document * stride..(document + 1) * stride];
                            // Left padding puts the document at the end of its slot, right at the start.
                            let (text, padding, at) = match params.direction {
                                PaddingDirection::Right => {
                                    let (text, padding) = slot.split_at_mut(len);
                                    (text, padding, document * stride)
                                }
                                PaddingDirection::Left => {
                                    let (padding, text) = slot.split_at_mut(stride - len);
                                    (text, padding, document * stride + stride - len)
                                }
                            };
                            // SAFETY: `text` is `len` uninitialised slots from `split_at_mut`, disjoint
                            // from every other job's region and from the source in this part's arena.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    arena[read..].as_ptr(),
                                    text.as_mut_ptr().cast::<PipelineToken>(),
                                    len,
                                );
                            }
                            for slot in padding.iter_mut() {
                                slot.write(pad);
                            }
                            mask_dst[at..at + len].fill(1);
                            read += len;
                        }
                    });
            }
            // SAFETY: every slot belonged to exactly one job, and each job wrote its documents and
            // filled the rest of their slots with the pad id.
            unsafe { ids.set_len(total) };
            return Ok(Encoding::padded(
                ids,
                mask,
                (0..=total_rows).map(|i| (i * stride) as u32).collect(),
            ));
        }

        let mut ids: Vec<PipelineToken> = Vec::with_capacity(total_ids);
        let mut offsets = vec![0u32; total_rows + 1];
        {
            let mut id_rest = &mut ids.spare_capacity_mut()[..total_ids];
            let mut off_rest = offsets.as_mut_slice();
            let mut base = 0u32;
            let mut jobs = Vec::with_capacity(parts.len());
            for (arena, lens, _) in &parts {
                let (id_dst, rest) = id_rest.split_at_mut(arena.len());
                id_rest = rest;
                let (off_dst, rest) = off_rest.split_at_mut(lens.len());
                off_rest = rest;
                jobs.push((arena, lens, base, id_dst, off_dst));
                base += arena.len() as u32;
            }
            jobs.par_iter_mut()
                .for_each(|(arena, lens, base, id_dst, off_dst)| {
                    // SAFETY: `id_dst` is `arena.len()` uninitialised slots from `split_at_mut`, so it
                    // is disjoint from every other job's run and cannot overlap the source.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            arena.as_ptr(),
                            id_dst.as_mut_ptr().cast::<PipelineToken>(),
                            arena.len(),
                        );
                    }
                    let mut at = *base;
                    for (slot, len) in off_dst.iter_mut().zip(lens.iter()) {
                        *slot = at;
                        at += *len;
                    }
                });
        }
        // SAFETY: the loop above handed every one of the `total_ids` slots to exactly one job,
        // and each job filled its run in full.
        unsafe { ids.set_len(total_ids) };
        offsets[total_rows] = total_ids as u32;
        Ok(Encoding::batch(ids, offsets))
    });
    laid_out.map(Some)
}
