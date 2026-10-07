# BPE training engine

This engine trains a byte-pair encoding vocabulary from weighted words. It returns
a string-to-ID vocabulary, merge rules in rank order, and special tokens. It stores
token occurrences at fixed corpus coordinates and processes compatible rules together
to reduce repeated work while preserving the training rule order.

The public entry points are in [the BPE trainer](../mod.rs). This directory owns
vocabulary construction, pair selection, corpus updates, and count publication.
[DESIGN.md](DESIGN.md) explains the algorithms and their safety contracts.

## How training works

A word's weight contributes to each adjacent token pair. For example, the word
`abab` with weight 3 contributes 6 to `(a, b)` and 3 to `(b, a)`. In the ordinary
first-activation mode, the engine selects the pair with the largest positive
count. Equal counts are ordered by the token-ID pair, with the smaller pair first.
Applying a rule merges nonoverlapping matches from left to right.

For the `abab` example, assume plain character tokens, no reserved merge result,
and a birth limit that admits the new boundary. Selecting `(a, b) -> ab` leaves
two `ab` tokens. The old `(b, a)` boundary disappears, and `(ab, ab)` is born with
count 3. The index records these count changes and the new boundary's coordinate
so the next selection can proceed without recounting every word.

One training call performs these steps:

1. Initialize the vocabulary from special tokens, the alphabet, and configured
   affixes. Borrow the weighted words and arrange them by descending weight.
2. Validate input bounds, count initial pairs, and build their occurrence lists.
   Then allocate the mutable corpus. Each retained input symbol has a fixed slot;
   merges update token endpoints without shifting the rest of a word.
3. Select an ordered prefix of compatible rules. Read the current corpus to
   prepare disjoint writes and neighboring pair changes. Apply the writes, then
   update counts and publish new occurrence lists. Each parallel phase joins
   before the next phase starts.
4. Repeat until the target vocabulary size is reached or the selection queue has
   no eligible rule. Release training storage and construct the public result.

If initialization already reaches the target size, the engine returns without
allocating mutable corpus slots. Numeric checks still apply: a safe total-mass
bound can prove that all initial pair counts fit; otherwise the engine counts
individual pairs before returning.

An ID is active once it has represented an input symbol or a selected merge
result. Within an attempt, assigned IDs stay fixed. A merged string already in the
vocabulary keeps its ID. If fresh-mode selection encounters an already active
result ID, the engine discards the attempt and restarts from the same input and
chosen alphabet in active-reuse mode. That mode selects one occurrence cohort at
a time and preserves its signed count-ledger and queue behavior. See
[token identity](DESIGN.md#token-identity-and-initialization)
and [active reuse](DESIGN.md#active-id-reuse).

## Public entry points and threads

- `BpeTrainer::do_train(&word_counts)` borrows a caller-owned weighted-word map
  and returns vocabulary entries, ordered merges, and special tokens.
- `BpeTrainer::train_vocab()` trains the trainer's stored counts and returns the
  same parts.
- `Trainer::train(&mut model)` trains the stored counts and replaces the BPE model.
- `Trainer::feed(iterator, process)` applies the caller's preprocessing callback
  and collects the words used by the next `train` or `train_vocab` call.

`feed` is outside the training core. Sequential feed retains one count map.
Parallel feed runs preprocessing behind the iterator bridge, accumulates local
counts into a shared table, and consumes the table into unique, unordered entries.
Both forms expose a borrowed `WordCountsView`; training leaves the stored counts
intact. Feed does not sort them. Trainer serialization remains a flat word-count
map, and equality compares counts independently of storage representation.

Feed uses the ambient Rayon pool, including a pool installed by the caller.
Training creates one pool per call and takes its worker count from
`tk_encode::parallelism::num_threads()`. Both respect the parallelism switch.
Disabling parallelism makes training use one worker; sequential feed executes on
the calling thread, which may still be inside a multiworker ambient pool.
Ordinary callback errors do not stop later feed callbacks. A failed feed keeps
the trainer's previous counts.

## Behavior to preserve

The engine's contracts include weighted counts, pair-ID tie breaks,
left-to-right overlap selection, affix interpretation, reserved IDs, and active-ID
reuse. The following boundaries matter when changing the implementation:

- Each training call builds a new vocabulary. ID stability applies within that
  attempt; it does not preserve IDs from an earlier model.
- Pair tie breaks are fixed. Equal-frequency alphabet truncation has no complete
  codepoint tie break, and decorated IDs depend on input traversal. Neither map
  traversal nor parallel feed entry order is stable. Worker parity for one input
  view does not imply identical output across arbitrary hash seeds.
- Engine pair, birth, and removal arithmetic is checked in `u64`. Reuse ledger
  updates are checked in `i64`. Nonempty affixes enforce signed input bounds even
  before reuse occurs. Feed and limited-alphabet accumulation use ordinary
  additions. See [numeric domains](DESIGN.md#numeric-domains).
- `max_token_length` is a strict admission limit for newborn pair spans in retained
  corpus slots. Initial pairs bypass it. See [the length rule](DESIGN.md#the-birth-length-rule).
- Training errors discard the attempt. A commit can fail after corpus writes and
  partial count updates; rounds have no rollback.

## Tests

Run from the repository root:

```sh
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --lib
cargo test --manifest-path tokenizers/tk-train/Cargo.toml --no-default-features --lib
```

The [engine tests](tests/mod.rs) compare vocabulary IDs, ordered merges, and
per-rule traces across worker counts. They cover ties, AA overlaps, affixes,
filtering, identity reuse, birth limits, overflow, and long words. The
[public contract tests](tests/public_contract.rs) cover feed, reload, thread
policy, progress, failures, and zero-merge results. Isolated thread-policy cases
use different ambient and training pool sizes and observe executing training tasks.

The small-input reference checks queue and cohort behavior independently of the
engine's storage. Explicit expectations cover wide counts beyond the reference's
count domain. Local index and storage tests cover ordered updates, complete and
partial publication, full-width positions, restart cursors, failed appends, arena
lifetimes, and scratch recovery. These are correctness checks; performance
claims require separate measurements.

## Measurement boundaries

For measurements, record source and binary hashes, input hashes, compiler and
features, worker count, CPU affinity, and raw repeated results. Report wall time,
CPU time, process high-water memory, and exact output validation. Measure `feed`,
public `do_train`, and end-to-end execution separately. Keep workload-specific
results in benchmark records.

## Source guide

- Start with [mod.rs](mod.rs): `Training` initializes and restarts attempts,
  manages their storage, and coordinates joined rounds. [batch.rs](batch.rs)
  owns rule selection.
- Read [vocabulary.rs](vocabulary.rs) and [corpus](corpus/mod.rs) for token
  identity and fixed coordinates. [Corpus planning](corpus/prepare.rs) hides
  symbol measurement and seek anchors in `WordMeasure`.
- Read [initial pairs](initial_pairs.rs) for `InitialCollector`, which selects
  the record layout and shares wave encoding and publication across collectors.
- Read [merge preparation](merge/prepare/mod.rs), [application](merge/mod.rs),
  and [owner commit](pair_index/commit.rs) for the read, write, and publication
  phases. [PairIndex](pair_index.rs) owns counts, cohorts, and queues.
- Read [storage](storage/mod.rs) and [execution](execution.rs) for compressed
  positions, allocation, and reusable worker resources. Local unsafe operations
  state their safety requirements next to the code.
