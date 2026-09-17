//! The parallel batch encode: one contiguous id buffer, nothing shared between workers.

use crate::parallelism::pool;
use crate::pipeline::{Document, Encoding, PipelineToken, PipelineTokenizer, Template};

use super::Result;

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
    inputs: &[&str],
    template: &Template,
    add_special_tokens: bool,
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
    let total: usize = inputs.iter().map(|s| s.len()).sum();
    let avg = (total / inputs.len().max(1)).max(1);
    // Enough documents to be worth a task -- twice the parallel floor measured best, below which
    // rayon's per-task cost starts to show -- but never so many that a thread gets only one run.
    let by_bytes = (2 * PARALLEL_MIN_BYTES).div_ceil(avg);
    let by_balance = (inputs.len() / (threads * 4).max(1)).max(1);
    let chunk = by_bytes.min(by_balance).max(1);

    let parts: Vec<Result<(Vec<PipelineToken>, Vec<u32>)>> = pool.install(|| {
        inputs
            .par_chunks(chunk)
            // One scratch per worker, not one per chunk. `ScratchPool` is a mutex, and the
            // scratch it hands back carries a 2 MB word cache: taking one per chunk both
            // contends on the lock and drags that cache between cores as the pool reissues
            // it to whichever thread asks next. Eight independent processes doing this same
            // work scale 7.8x; taking the scratch per chunk was most of what stood between
            // that and the pool.
            .map_init(
                || tok.scratch(),
                |scratch, docs| {
                    let bytes: usize = docs.iter().map(|s| s.len()).sum();
                    let mut arena =
                        Vec::with_capacity(bytes / 4 + template.n_special() * docs.len());
                    let mut lens = Vec::with_capacity(docs.len());
                    for doc in docs {
                        let start = arena.len();
                        tok.frame(
                            Document::from(*doc),
                            template,
                            add_special_tokens,
                            scratch,
                            &mut arena,
                            None,
                        )?;
                        lens.push((arena.len() - start) as u32);
                    }
                    Ok((arena, lens))
                },
            )
            .collect()
    });
    let parts = parts.into_iter().collect::<Result<Vec<_>>>()?;

    // Concatenating the arenas in input order is the whole serial tail, and at eight threads it
    // was most of what stopped this scaling: fitting Amdahl to the measured curve put the serial
    // fraction at 7.5%, which is what a memcpy of every id and a push per row costs. Each part's
    // destination is known once the lengths are, though, so the copy can be handed back to the
    // pool -- `split_at_mut` hands out the disjoint runs and the workers fill them at once.
    let total_ids: usize = parts.iter().map(|(arena, _)| arena.len()).sum();
    let total_rows: usize = parts.iter().map(|(_, lens)| lens.len()).sum();

    // `vec![PipelineToken(0); n]` would zero three megabytes the workers are about to
    // overwrite, and that memset is serial: it cost more than the parallel copy saved.
    // Hand out the uninitialised capacity instead -- `split_at_mut` still proves the runs
    // are disjoint, so the only thing left unchecked is that each is fully written, which
    // it is: every part copies exactly its own length.
    let mut ids: Vec<PipelineToken> = Vec::with_capacity(total_ids);
    let mut offsets = vec![0u32; total_rows + 1];
    {
        let mut id_rest = &mut ids.spare_capacity_mut()[..total_ids];
        let mut off_rest = offsets.as_mut_slice();
        let mut base = 0u32;
        let mut jobs = Vec::with_capacity(parts.len());
        for (arena, lens) in &parts {
            let (id_dst, rest) = id_rest.split_at_mut(arena.len());
            id_rest = rest;
            let (off_dst, rest) = off_rest.split_at_mut(lens.len());
            off_rest = rest;
            jobs.push((arena, lens, base, id_dst, off_dst));
            base += arena.len() as u32;
        }
        pool.install(|| {
            jobs.par_iter_mut()
                .for_each(|(arena, lens, base, id_dst, off_dst)| {
                    // SAFETY: `id_dst` is `arena.len()` uninitialised slots handed out by
                    // `split_at_mut`, so it is disjoint from every other job's run and cannot
                    // overlap the source, which lives in the part.
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
        });
    }
    // SAFETY: the loop above handed every one of the `total_ids` slots to exactly one job,
    // and each job filled its run in full.
    unsafe { ids.set_len(total_ids) };
    offsets[total_rows] = total_ids as u32;
    Ok(Some(Encoding::batch(ids, offsets)))
}
