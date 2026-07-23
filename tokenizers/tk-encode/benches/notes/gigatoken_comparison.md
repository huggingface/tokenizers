# gigatoken vs tk-encode — per-step profile + what to take

Branched off #2223 (`feat/bpe-cache`). gigatoken = marcelroed/gigatoken @ 0.9.0, a
bulk-throughput / training tokenizer (tuned on 1 GB OpenWebText, Zen). Reproduction harness:
`notes/gigatoken_profile_steps.rs` (drop into a gigatoken clone as `examples/profile_steps.rs`,
`cargo run --release --example profile_steps`). Decomposes the encode pipeline through gigatoken's
public API — `pretokenize` iterator (split), `memoized_encode_flat` (split+probe, no added-split),
`encode_with_added_tokens_flat` (adds the added-token matcher), cold-vs-warm isolates merge.

## Per-step profile (ns/byte, 1st-touch corpora, tokenizer load excluded)

```
tokenizer    lang        split   probe   added   merge |    warm    cold
gpt2         English     0.730   0.105   0.029   1.545 |   0.863   2.408
gpt2         Russian     1.210   1.399   0.146   5.765 |   2.756   8.521
gpt2         Chinese     1.864   0.781   0.115  11.050 |   2.759  13.809
gpt2         Korean      1.665   1.288   0.028   4.919 |   2.982   7.901
deepseek-v4  English     1.518   0.211   0.000   1.672 |   1.727   3.399
deepseek-v4  Chinese     1.018   0.669   2.490  23.035 |   4.177  27.213
deepseek-v4  Korean      1.640   0.842   3.054  16.815 |   5.536  22.351
llama-3      English     0.802   0.187   0.000   2.747 |   0.976   3.724
llama-3      Chinese     1.898   0.555   0.156  25.243 |   2.609  27.852
llama-3      Korean      1.821   0.488   0.111  17.059 |   2.420  19.479
```

- **split** = pretokenization (SIMD span-finding). **probe** = key hash + cache hit + emit.
  **added** = added-token matcher pass (aho-corasick over the segment; ~0 with no special-token
  hits, rises on deepseek CJK where the matcher does more work). **merge** = cache-miss BPE
  (cold − warm). **warm** = full encode, all cache hits. **cold** = first touch, all misses.

## The one finding that matters

**Merge dominates by an order of magnitude; everything else is noise.** On a cold / low-hit-rate
workload, merge is 1.5–25 ns/B while split+probe+added together are 0.8–4 ns/B. On a warm workload
(the cache's whole point) the total collapses to 0.86–5.7 ns/B — i.e. the pretoken cache is worth
5–50× and is the entire throughput story on repetitive input.

Strategic implication for us: the levers are (1) cache hit rate and (2) merge speed on misses.
Split and added-split are already sub-2 ns/B in both codebases and not worth further work.
Normalization (our big atomnorm effort) is upstream of the cache key and runs every byte
regardless of hit — it matters for the *warm* path (where it's a real fraction of the few ns/B),
but it cannot touch the merge-bound cold path.

## Step by step — what we have vs gigatoken

| step | tk-encode (this branch) | gigatoken | verdict |
|---|---|---|---|
| normalize | atomnorm scan + prefilter, sub-1 ns/B, spm charsmap prefiltered 10–45× | `spm_precompiled` raw + `icu`, no prefilter | **we win** — keep ours |
| split | atomsplit SIMD classify + FSM | fused SIMD pretokenizers + CRC key hash + prefetch, per-family | comparable; their fused key-hash-in-walk is a real idea (below) |
| added-split | matcher | aho-corasick, fused into for_each_piece | comparable, both ~0 on normal text |
| word cache | `WordCache`: 4-way set-assoc, bucket=1 line, SoA arenas, tag filter, skip-on-full | `ShortPretokenCache`: open-addr pair-probe, inline value pack, grows to DRAM, huge pages, prefetch ladder | **different regime — see below** |
| merge | hand-rolled binary u64 heap (`MERGE_HEAP_MIN=24`) | `PairRankTable` (dense array for hot IDs + flat open-addr table) + NEON merge core | **worth a head-to-head** |

## The cache regime distinction (the crux)

Our `WordCache` and gigatoken's `ShortPretokenCache` solve the same problem in opposite regimes:

- **Ours (inference):** bounded, fixed bucket count, **skip on full bucket** — accept a miss rather
  than grow. Bucket is `#[repr(align(64))]` = one cache line, 4 ways scanned branch-lessly. Stays
  L2-resident. Correct for encoding a document / request where the working set is small.
- **gigatoken (training):** unbounded, grows at 3/4 load to hold ~1.3M unique pretokens (~99.4%
  hit) far past L3 — so every tail lookup is a DRAM access, which is why it needs 2 MiB huge pages
  (`MADV_HUGEPAGE`, dTLB), a two-stage prefetch ladder (L2 a chunk ahead, L1 a few probes ahead),
  and a branchless probe that touches exactly one line. Correct for streaming 1 GB once.

Their huge-page / prefetch machinery only pays when the table is >64 MB. For our workload it's dead
weight — the bounded L2 cache is the right call, as designed.

## Take / have / skip

**Worth taking:**
1. **Inline value packing.** 90% of pretokens encode to 1 token, 98% to ≤2 — gigatoken packs up to
   4 token ids *inline* in the slot (`u64 val + u64 ext`), so a hit is one dependent load with no
   second random access into an ids arena. Our `WordCache` stores `(ids_off, ids_len)` into a side
   `ids` Vec — a second (usually cache-cold) access. Packing ≤2 ids inline in the slot would remove
   it for ~98% of hits. **Highest-value, small change.**
2. **PairRankTable for the merge path.** A dense direct-index array for the hot ~1.8k merged IDs
   (round-1 + most mid-merge lookups: one load, no hash) backed by a flat open-addressed table for
   the rest, replacing the hashbrown merges map. Directly comparable to our heap+VocabStore merge;
   given merge is the dominant cost, **bench it head-to-head** — it may beat our heap on the
   miss-heavy path.
3. **Fused key-hash-in-the-span-walk.** gigatoken derives the cache key hash *inside* the
   pretokenizer's chunk fill (`fill_spans_keyed`) so the byte scan and hash share one pass. Our
   split and cache-key hash are separate passes. Minor, but free once the walker exists.

**Already have (equal or better):**
- Normalization (atomnorm strictly ahead), split (atomsplit comparable), the 4-way L2 WordCache
  (correct for our regime — do **not** replace with their DRAM design), merge heap (competitive;
  see PairRankTable bench above before deciding).

**Not necessary for us:**
- Huge pages / `MADV_HUGEPAGE`, the L2/L1 prefetch ladder, DRAM-resident open-addressing growth —
  all only pay above ~64 MB (1 GB-corpus training), irrelevant to inference-sized working sets.
- The per-arch asm `csel`/`cmov` branchless probe — a micro-opt for a DRAM-latency-bound probe;
  our probe hits L2, so take only if a profile shows it on the critical path.
- parquet / jsonl / zstd input, batch file encoding, huge-page allocator, py bindings — out of scope.
```

---

## UPDATE — fair per-step head-to-head vs PR #2190 (cached "ours")

The verdict above was too generous and used the *uncached* pipeline. Re-run with PR #2190
(`poc-merge-cache`: FlatCache + MPHF RankStore) as "ours", one process, ids byte-identical to
gigatoken on every row (`idsEq OK`). ns/byte, aligned steps (harness:
`notes/gigatoken_vs_ours_per_step.rs`, needs the 2190 worktree + a gigatoken clone, nightly).

```
tok         lang       | o.nrm o.splt o.mdlW o.mdlC | t.splt t.prb  t.mrg
gpt2        English    |  0.00   1.79   3.59   8.66 |  0.73   0.11   1.37
gpt2        Russian    |  0.00   1.28   2.17  12.38 |  1.21   1.38   6.26
gpt2        Chinese    |  0.00   1.27   1.46  15.89 |  1.94   0.70  11.54
gpt2        Korean     |  0.00   1.29   2.60   9.33 |  1.67   1.11   4.80
llama-3     English    |  0.00   1.90   3.07   6.54 |  0.81   0.17   2.43
llama-3     Chinese    |  0.00   1.21   1.12  37.72 |  1.98   0.37  25.06
deepseek-v4 English    |  0.00   3.25   3.40   8.40 |  1.50   0.20   1.74
deepseek-v4 Chinese    |  0.00   1.46   1.28  30.76 |  0.98   0.54  23.34
```
o.* = ours marginals (nrm/split/model-warm/model-cold); t.* = theirs (split / probe+emit / merge).

**Which part wins, quantified:**
1. **Warm emit is the dominant gap — theirs up to 33× (English 3.59 vs 0.11), 1.6–2.3× elsewhere.**
   Their branchless `probe_pair` + inline value packing + flat u32 store-and-advance is memcpy-tier;
   ours does a FlatCache `get` + copy per pretoken. **Highest-value fix by far, and it's the emit
   path, not merge.**
2. **Merge (cold): theirs 1.3–6× faster** (English 6×, CJK 1.3–1.5×). PairRankTable + NEON core.
   Corrects the earlier "merge competitive" claim — it is not.
3. **Split: ours 2–2.5× slower ONLY on whitespace-heavy Latin** (GPT contraction regex); ours is
   comparable or faster on CJK/Cyrillic. gigatoken has per-family hand-fused SIMD English splitters.
4. **Normalize = 0** for GPT-2/Llama/DeepSeek (byte-level BPE, no normalizer) — atomnorm doesn't
   execute for these models; it only pays for BERT/T5/SPM.

Revised priority for closing the gap: warm emit (flat u32 out + inline-packed ids + branchless
probe) >> merge (PairRankTable/NEON) > Latin split. Normalization is not on the critical path here.
