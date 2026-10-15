mod ascii_fastpath;
mod character_cache;
mod rewriter;

pub(crate) use ascii_fastpath::unstable_spans_iter;
pub(crate) use character_cache::CharacterCache;
pub(crate) use rewriter::Rewriter;

/// `count` texts of up to `max_len` bytes, each a random concatenation of `pieces`.
#[cfg(test)]
pub(crate) fn random_texts(
    seed: u64,
    pieces: &[&str],
    count: usize,
    max_len: usize,
) -> Vec<String> {
    let mut state = seed;
    let mut next = |bound: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % bound
    };
    (0..count)
        .map(|_| {
            let len = next(max_len);
            let mut text = String::new();
            while text.len() < len {
                text.push_str(pieces[next(pieces.len())]);
            }
            text
        })
        .collect()
}
