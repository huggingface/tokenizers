//! Batch encode: documents in, one contiguous id buffer out.
//!
//!   documents  [d0 d1 d2 d3 d4 d5 ..................................... dN]
//!                   |  par_chunks: a run of documents per task
//!        +----------+----------+----------+----------+
//!     worker 0   worker 1   worker 2   worker 3            one scratch each,
//!        |          |          |          |                reused across tasks
//!     arena 0    arena 1    arena 2    arena 3             private, stays in cache
//!        +----------+----+-----+----------+
//!                        |  split_at_mut: disjoint runs, filled in parallel
//!                        v
//!   unpadded   ids [d0|d1|d2|d3|d4|.....................]  document i is
//!              offsets  ^  ^  ^  ^                         ids[offsets[i]..offsets[i+1]]
//!
//!   padded     ids [d0...pad|d1.....pad|d2......pad|....]  every document `stride` wide,
//!                   <- stride ->                           so it reads as a 2D array
//!
//! The arenas are the point: a worker writing into the shared output instead measured slower,
//! because every write then lands in a multi-megabyte buffer and allocates a cache line.
use crate::parallelism::pool;
use crate::pipeline::{Document, Encoding, PipelineToken, PipelineTokenizer, Template};
use crate::utils::padding::{PaddingDirection, PaddingParams, pad_length};

use super::Result;

/// What one worker hands back: its arena of ids, the length of each document in it, and the
/// longest of those, which is the stride when the batch is padded.
type Part = (Vec<PipelineToken>, Vec<u32>, u32);

/// Below this many bytes a batch is encoded serially: the pool costs more than it saves.
///
/// `pub` so the differential tests in `tk-convert` can size an input past it.
pub const PARALLEL_MIN_BYTES: usize = 8 * 1024;

/// One thread runs the same flat assembly serially, so this is one algorithm at every width.
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

    let total: usize = inputs.iter().map(|d| d.text.len()).sum();
    let avg = (total / inputs.len().max(1)).max(1);
    let by_bytes = (2 * PARALLEL_MIN_BYTES).div_ceil(avg);
    let by_balance = (inputs.len() / (threads * 4).max(1)).max(1);
    let chunk = by_bytes.min(by_balance).max(1);

    let laid_out = pool.install(|| -> Result<Encoding> {
        let parts: Vec<Result<Part>> = inputs
            .par_chunks(chunk)
            // One scratch per worker, not per chunk: the 2 MB word cache must not migrate.
            .map_init(
                || tok.scratch(),
                |scratch, docs| {
                    let bytes: usize = docs.iter().map(|d| d.text.len()).sum();
                    // Three bytes a token, not four: a quarter left every arena one realloc short.
                    let mut arena =
                        Vec::with_capacity(bytes / 3 + template.n_special() * docs.len() + 16);
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
            // Workers track their own longest, so the stride costs one value per chunk.
            let longest = parts.iter().map(|(.., l)| *l).max().unwrap_or(0) as usize;
            let stride = pad_length(longest, params).max(longest);
            let total = total_rows * stride;
            let pad = PipelineToken::from(params.pad_id);
            let mut ids: Vec<PipelineToken> = Vec::with_capacity(total);
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
                            // SAFETY: `text` is `len` uninitialised slots from `split_at_mut`, disjoint from every other job.
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
                    // SAFETY: `id_dst` is `arena.len()` uninitialised slots from `split_at_mut`, disjoint per job.
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
        unsafe { ids.set_len(total_ids) };
        offsets[total_rows] = total_ids as u32;
        Ok(Encoding::batch(ids, offsets))
    });
    laid_out.map(Some)
}
