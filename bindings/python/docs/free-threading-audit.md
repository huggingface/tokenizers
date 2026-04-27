# Free-threaded Python (3.14t) — concurrency audit

This document captures the audit done before lifting the
`#[cfg(not(Py_GIL_DISABLED))]` setter freeze that 0.23.0 shipped. It
exists so reviewers can cross-check the reasoning rather than take
"it should be fine" on faith, and so future maintainers can extend
the audit when new mutable surface is added.

## Scope

Everything in `bindings/python/src/{decoders,models,normalizers,pre_tokenizers,processors,tokenizer,trainers}.rs`
that's exposed to Python. Specifically, every `#[setter]` and every
mutation path reachable from Python code under free-threaded CPython.

## Synchronization primitives in play

Every component wrapper is shaped like:

```rust
Arc<RwLock<Wrapper>>          // single component
Vec<Arc<RwLock<Wrapper>>>     // sequence of components
```

- `Arc` is reference-counted; cloning is atomic and Rust guarantees
  no use-after-free.
- `std::sync::RwLock` from std serializes writers and admits multiple
  concurrent readers. Lock acquisition is the synchronization point.
- PyO3 free-threaded mode additionally provides per-object locking
  for `&mut self` methods (we don't use `&mut self` here — all our
  setters take `PyRef<Self>` and route through the inner `RwLock`).

## Audit categories

### 1. Single-field setter (e.g. `bpe_trainer.vocab_size = X`)

Path: `setter!` macro → `wrap.write().unwrap()` → field write.

**Concurrent reader semantics**: writer blocks until all readers drop
their guards. Reader sees either old or new value, never torn.
**Concurrent writer semantics**: writers serialize through the
`RwLock`.

**Verdict: SAFE for component-internal setters.** Standard `RwLock`
discipline; no shared `&mut` reaches outside the guarded scope.
Confirmed by `test_encode_while_mutating_trainer_fields` and the
trainer mutation test in `tests/test_freethreaded.py`.

### 2. Top-level component swap (e.g. `tokenizer.post_processor = X`)

Path: setter on `PyTokenizer` → replaces the wrapped `Tokenizer`'s
relevant component.

**Verdict: SAFE.** `PyTokenizer` now wraps the inner tokenizer in a
`std::sync::RwLock<Tokenizer>`. Every method that mutates the
tokenizer takes `&self` and acquires the write guard; readers acquire
the read guard. Concurrent setters serialize through `RwLock` instead
of racing PyO3's per-pyclass borrow check.

Confirmed by `test_encode_while_swapping_post_processor` and
`test_concurrent_setters_no_lock_poisoning` in
`tests/test_freethreaded.py`.

**Historical note**: 0.23.0 shipped with `&mut self` setters and a
freeze (every setter cfg-gated out under `Py_GIL_DISABLED`). That
worked but cost users the ability to mutate tokenizers post-construct
on 3.14t. 0.23.2's `RwLock<Tokenizer>` rewrite removed the freeze.

### 3. Compound mutation (e.g. `tokenizer.post_processor.special_tokens = X`)

Python evaluates this in two steps:

1. `tokenizer.post_processor` → returns a Py wrapper holding a clone
   of `Arc<RwLock<...>>`.
2. `obj.special_tokens = X` → setter on that wrapper acquires the
   inner `RwLock`'s write guard and mutates.

Under free-threading, another thread can call `tokenizer.post_processor = Y`
between steps 1 and 2. Step 2 still mutates the **old** wrapper
(now orphaned).

**Verdict: BEHAVIORAL SURPRISE, NOT A DATA RACE.** The mutation is
"lost" — the new post_processor is untouched. This is the same class
of race as `d[k] = v` racing with `d.clear()` in stdlib; Python
documents that users coordinate via locks for compound operations.

**Mitigation**: documented in README under the 3.14t section. No code
change needed.

### 4. Sequence components (`Sequence(Vec<Arc<RwLock<Wrapper>>>)`)

The `Vec` itself is owned by an outer wrapper; mutating the sequence
goes through the outer wrapper's `RwLock`. Each inner element has its
own `RwLock` for per-element mutation.

**Verdict: SAFE for individual element access.** Iteration order is
stable while a read guard is held on the outer wrapper.

### 5. Trainer mutation during `train()`

`train()` is called on the tokenizer, which holds the trainer through
an `Arc<RwLock<...>>`. The train loop in
`tokenizers/src/models/{bpe,wordpiece,wordlevel,unigram}/trainer.rs`
clones the relevant config fields once at the top of the function and
operates on the clone.

A concurrent `bpe_trainer.vocab_size = X` after `train()` started will
mutate the trainer wrapper but **not** affect the in-flight train run,
because the train loop already cloned `vocab_size`. This matches GIL
semantics (mutation between two atomic Python statements).

**Verdict: SAFE.** Verified by reading the train loop sources; the
config is not re-read mid-pass.

### 6. Tokenizer fast paths during `encode()` / `encode_batch()`

`encode()` takes a read guard on each component (`normalizer`,
`pre_tokenizer`, `model`, `post_processor`, `decoder`) for the
duration it operates on that component. A concurrent setter waits
until all in-flight encodes drop their read guards.

This serializes writes but reads are unbounded-concurrent (the point
of `RwLock`), so encode throughput scales with cores.

**Verdict: SAFE; performance acceptable.** Verified that no encode
path holds a read guard across a `Python::with_gil` boundary or any
other suspension point.

## Things that would break this audit

If any of these are introduced, re-audit before merging:

- A method that takes `&mut self` on a `PyTokenizer` / component
  (PyO3 will provide per-object locking but it interacts oddly with
  the inner `RwLock`).
- A getter that returns a `&` reference into the inner Wrapper (would
  need the read guard to outlive the returned reference; PyO3
  doesn't allow that today, but custom impls could).
- A long-running operation that releases the read guard mid-flight
  (e.g. `encode_batch` parallelizing internally and re-acquiring).
  Currently `encode_batch` holds the guard for the whole batch.
- New `parking_lot` or `tokio::sync` locks introduced anywhere that
  wrap component state — they don't compose cleanly with `std::sync::RwLock`
  poisoning semantics.

## Stress test

`tests/test_freethreaded.py` runs N encoder threads against M setter
threads on the same `Tokenizer` instance and asserts:

1. No panic / SystemError.
2. No `RwLock` poisoning (would manifest as `PyException("RwLock
   synchronisation primitive is poisoned")`).
3. Encode results are valid for one of the configurations seen during
   the run (compound-mutation surprise from §3 is allowed).

Run it under both regular CPython and 3.14t — both must pass.
