//! Local preaggregation into a shared table; feed returns unordered counts.
use super::word_counts::WordCounts;
use ahash::{AHashMap, RandomState};
use compact_str::CompactString;
use std::collections::hash_map::Entry;
use tk_encode::{Result, parallelism::*};

type CountMap = AHashMap<CompactString, u64>;
type SharedCounts = scc::HashMap<CompactString, u64, RandomState>;
// Cap distinct local keys; strings and callback output are not byte-bounded.
const LOCAL_KEY_LIMIT: usize = 2048;
// Amortize the bridge's serialized source `next()` lock across several inputs.
const FEED_BATCH_SIZE: usize = 32;

fn batches<I, S>(iterator: I, mut startup_singles: usize) -> impl Iterator<Item = Vec<S>> + Send
where
    I: Iterator<Item = S> + Send,
    S: Send,
{
    // Keep one exhaustion state across batches, including a partial final batch.
    let mut iterator = iterator.fuse();
    std::iter::from_fn(move || {
        if startup_singles > 0 {
            let item = iterator.next()?;
            startup_singles -= 1;
            return Some(vec![item]);
        }
        let batch: Vec<_> = iterator.by_ref().take(FEED_BATCH_SIZE).collect();
        (!batch.is_empty()).then_some(batch)
    })
}

fn flush(local: &mut CountMap, shared: &SharedCounts) {
    for (word, count) in local.drain() {
        shared
            .entry_sync(word)
            .and_modify(|total| *total += count)
            .or_insert(count);
    }
}

fn add_local(local: &mut CountMap, word: CompactString, shared: &SharedCounts) {
    match local.entry(word) {
        Entry::Occupied(mut entry) => *entry.get_mut() += 1,
        Entry::Vacant(entry) => {
            entry.insert(1);
            // Only a new key can fill the cache; repeated words need no size check.
            if local.len() == LOCAL_KEY_LIMIT {
                flush(local, shared);
            }
        }
    }
}

fn accumulate<S, F, A>(
    counts: Result<CountMap>,
    sequence: S,
    process: &F,
    mut add: A,
) -> Result<CountMap>
where
    S: AsRef<str>,
    F: Fn(&str) -> Result<Vec<String>>,
    A: FnMut(&mut CountMap, CompactString),
{
    // Run the callback even after a normal error; retain this fold's first error.
    let words = process(sequence.as_ref());
    let mut counts = counts?;
    for word in words? {
        add(&mut counts, CompactString::from(word));
    }
    Ok(counts)
}

pub(super) fn count<I, S, F>(iterator: I, process: &F) -> Result<WordCounts>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
    F: Fn(&str) -> Result<Vec<String>> + Sync,
{
    let hash = RandomState::default();
    let new_counts = || Ok(CountMap::with_hasher(hash.clone()));
    let parallel_workers = if get_parallelism() {
        current_num_threads()
    } else {
        1
    };
    if parallel_workers == 1 {
        // A single worker needs neither shared updates nor a final copy. Keep
        // its map, using the same callback/error handling as the parallel path.
        return iterator
            .fold(new_counts(), |counts, sequence| {
                accumulate(counts, sequence, process, |counts, word| {
                    *counts.entry(word).or_default() += 1;
                })
            })
            .map(WordCounts::from_map);
    }
    let shared = SharedCounts::with_hasher(hash.clone());
    // Keep preprocessing behind the upstream bridge: mapping it on the input
    // iterator would execute callbacks under the bridge's serial next() lock.
    // Local caches absorb repeats; flushes update weighted counts directly,
    // avoiding a separate partition-and-reduce pass over all local entries.
    // Seed the ambient pool with independent inputs before amortizing the
    // bridge lock. A short stream of expensive documents would otherwise form
    // one batch whose callbacks all run serially on the same worker. Source
    // exhaustion naturally caps these singletons at the available input count.
    let results: Vec<Result<CountMap>> = batches(iterator, parallel_workers)
        .maybe_par_bridge()
        .flat_map_iter(std::iter::IntoIterator::into_iter)
        .fold(new_counts, |counts, sequence| {
            accumulate(counts, sequence, process, |counts, word| {
                add_local(counts, word, &shared)
            })
        })
        .collect();
    let locals: Vec<_> = results.into_iter().collect::<Result<Vec<_>>>()?;
    locals
        .into_maybe_par_iter()
        .for_each(|mut local| flush(&mut local, &shared));
    // Training only traverses counts, so consume keys into an unordered vector
    // instead of paying to allocate and rehash another dictionary. Training
    // sorts borrowed entries; feed retains no frequency ordering or index.
    let mut words = Vec::with_capacity(shared.len());
    // Keep table capacity stable while consumption removes its entries.
    let _reservation = shared.reserve(shared.capacity());
    shared.iter_mut_sync(|entry| {
        words.push(entry.consume());
        true
    });
    Ok(WordCounts::from_entries(words))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_are_lazy_ordered_and_keep_the_final_partial_batch() {
        let pulled = std::sync::atomic::AtomicUsize::new(0);
        let source = std::iter::from_fn(|| {
            let next = pulled.load(std::sync::atomic::Ordering::Relaxed);
            if next == FEED_BATCH_SIZE + 1 {
                None
            } else {
                pulled.store(next + 1, std::sync::atomic::Ordering::Relaxed);
                Some(next)
            }
        });
        let mut batches = batches(source, 0);

        assert_eq!(pulled.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(
            batches.next().unwrap(),
            (0..FEED_BATCH_SIZE).collect::<Vec<_>>()
        );
        assert_eq!(
            pulled.load(std::sync::atomic::Ordering::Relaxed),
            FEED_BATCH_SIZE
        );
        assert_eq!(batches.next().unwrap(), vec![FEED_BATCH_SIZE]);
        assert_eq!(
            pulled.load(std::sync::atomic::Ordering::Relaxed),
            FEED_BATCH_SIZE + 1
        );
        assert_eq!(batches.next(), None);
    }

    #[test]
    fn startup_items_are_singletons_before_full_batches() {
        const STARTUP: usize = 4;
        let pulled = std::sync::atomic::AtomicUsize::new(0);
        let source = std::iter::from_fn(|| {
            let next = pulled.load(std::sync::atomic::Ordering::Relaxed);
            if next == STARTUP + FEED_BATCH_SIZE + 2 {
                None
            } else {
                pulled.store(next + 1, std::sync::atomic::Ordering::Relaxed);
                Some(next)
            }
        });
        let mut batches = batches(source, STARTUP);

        for index in 0..STARTUP {
            assert_eq!(batches.next(), Some(vec![index]));
            assert_eq!(pulled.load(std::sync::atomic::Ordering::Relaxed), index + 1);
        }
        assert_eq!(
            batches.next(),
            Some((STARTUP..STARTUP + FEED_BATCH_SIZE).collect())
        );
        assert_eq!(
            batches.next(),
            Some(vec![
                STARTUP + FEED_BATCH_SIZE,
                STARTUP + FEED_BATCH_SIZE + 1
            ])
        );
        assert_eq!(batches.next(), None);
    }

    #[test]
    fn batches_stop_at_first_none_even_when_the_source_resumes() {
        fn source(prefix: usize) -> impl Iterator<Item = usize> + Send {
            let mut step = 0;
            std::iter::from_fn(move || {
                let current = step;
                step += 1;
                if current < prefix {
                    Some(current)
                } else if current == prefix + 1 {
                    Some(usize::MAX)
                } else {
                    None
                }
            })
        }
        for startup in [0, 1, 4, FEED_BATCH_SIZE, 128] {
            for prefix in [0, 1, 3, 4, 5, 31, 32, 33, 35, 36, 37, 64, 65] {
                // Establish that this fixture really resumes after its first None.
                let mut raw = source(prefix);
                assert_eq!(raw.nth(prefix), None);
                assert_eq!(raw.next(), Some(usize::MAX));
                let mut grouped = batches(source(prefix), startup);
                let mut items = Vec::new();
                for batch in grouped.by_ref() {
                    items.extend(batch);
                }
                assert_eq!(
                    items,
                    (0..prefix).collect::<Vec<_>>(),
                    "startup={startup}, prefix={prefix}"
                );
                assert_eq!(grouped.next(), None);
                assert_eq!(grouped.next(), None);
            }
        }
    }

    #[test]
    fn weighted_flush_and_error_preserve_local_contracts() {
        let shared = SharedCounts::with_hasher(RandomState::default());
        let mut local = CountMap::default();
        for _ in 0..3 {
            add_local(&mut local, "word0".into(), &shared);
        }
        for i in 1..LOCAL_KEY_LIMIT {
            add_local(&mut local, format!("word{i}").into(), &shared);
        }
        assert!(local.is_empty());
        assert_eq!(shared.len(), LOCAL_KEY_LIMIT);
        assert_eq!(*shared.get_sync("word0").unwrap().get(), 3);
        add_local(&mut local, "word0".into(), &shared);
        add_local(&mut local, "tail".into(), &shared);
        flush(&mut local, &shared);
        assert!(local.is_empty());
        assert_eq!(*shared.get_sync("word0").unwrap().get(), 4);
        assert_eq!(*shared.get_sync("tail").unwrap().get(), 1);

        // This fold has already flushed before a callback fails. Later callbacks
        // run, but neither their output nor a later error replaces its first error.
        let mut state = Ok(local);
        let calls = std::cell::Cell::new(0);
        for sequence in ["first", "success", "second"] {
            state = accumulate(
                state,
                sequence,
                &|s| {
                    calls.set(calls.get() + 1);
                    match s {
                        "success" => Ok(vec!["discarded".into()]),
                        _ => Err(s.to_owned().into()),
                    }
                },
                |counts, word| add_local(counts, word, &shared),
            );
        }
        assert_eq!(state.unwrap_err().to_string(), "first");
        assert_eq!(calls.get(), 3);
        assert!(shared.get_sync("discarded").is_none());
        assert_eq!(shared.len(), LOCAL_KEY_LIMIT + 1);
    }
}
