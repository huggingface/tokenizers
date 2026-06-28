# `poc/` — fast, light, allocation-free BPE encode (research prototype)

A from-scratch, inference-only byte-level BPE encode path that is **14–18× faster than HuggingFace
`tokenizers` (main)** on the models it's byte-exact for, **~10× smaller**, **~3× lighter on RAM**, and
**allocation-free in the hot path** — plus a **56× multi-turn re-encode** prefix cache.

Context: Crusoe + NVIDIA Dynamo published *"Reducing TTFT by CPU-maxxing tokenization"* (Mar 2026)
showing tokenization is worth **up to 40% of TTFT**, and shipped **fastokens** (9.1× over HF) as a
**fork**. This POC explores capturing those wins — and going further — so they can live in the
**canonical `tokenizers` crate** instead of fragmenting across forks. Full write-up + the boss memo:
[`ENCODE_PERF_CASE.md`](./ENCODE_PERF_CASE.md).

## Layout

| crate | what it is |
|---|---|
| [`fast-encode/`](./fast-encode) | the final encoder + all benches (NEON-DFA split, MPHF VocabStore, alloc-free hybrid merge, thread-local pretoken cache, multi-turn prefix cache, IREE-style ring buffer) |
| [`edge-minimal/`](./edge-minimal) | the cleaned, C-free, allocation-free build for on-device — **0.37 MB** stripped binary (drops onig/pcre2/ahash/rayon) |
| [`special-token-matcher/`](./special-token-matcher) | added-token matching shootout: **MPHF length-probe (ours) vs IREE-style linear bucket scan vs daachorse Aho-Corasick** (speed + memory) |
| [`scripts/`](./scripts) | tokenizer downloader + the `apply_chat_template` workload generators |
| `data/toks/` | (gitignored) the 22 model `tokenizer.json` for the SWEEP — fetch with the script below |

Each crate is standalone (`cargo build --release` in its dir). Data paths resolve relative to the
crate, reusing the repo's `tokenizers/data/{llama-3-tokenizer.json,big.txt}`.

## Run the benches

```bash
# 1) Final encoder — parity, per-stage profile (split / bytes->id / merge / cache), ablation,
#    splitter shootout, parallel scaling, multi-turn prefix cache:
cd poc/fast-encode && cargo run --release

PROF_ONLY=1 cargo run --release   # just the DFA-path stage profile + alloc-free proof
CMP=1       cargo run --release   # cached vs uncached across 1/2/4/8 threads

# 2) Per-model x per-task SWEEP (needs the 22 tokenizers):
python ../scripts/download_tokenizers.py     # -> poc/data/toks/*.json
SWEEP=1 cargo run --release

# 3) Special-token matcher shootout (MPHF vs IREE vs daachorse):
cd ../special-token-matcher && cargo run --release
PROMPTS=$PWD/prompts_adv.json cargo run --release   # adversarial code / <|-spam set

# 4) Edge-minimal footprint:
cd ../edge-minimal && cargo run --release
ls -l target/release/edge-minimal   # ~0.37 MB stripped
```

## Headline results (Apple Silicon, llama-3, hot cache, ids-only unless noted)

| | HF main | fastokens (Crusoe/NVIDIA) | **POC** |
|---|--:|--:|--:|
| single-thread | ~6 MB/s | ~118 | **250–440** (cached, by workload) |
| 8-thread | ~50 | ~178 (35% scaling) | **~600–690** (95% scaling) |
| binary | 3.6 MB | 3.4 MB | **0.37 MB** (edge build) |
| heap (loaded) | 28 MB | 60 MB | **18.6 MB** |
| hot-path allocations | — | — | **0 (proven)** |
| multi-turn re-encode | full each turn | split cache | **56× (O(N²)→O(N))** |

**Matcher** (special-token matching on real prompts): MPHF length-probe is fastest on text
(0.65–0.79 ns/byte) and **~free in memory** (~138 B/bucket, reuses the VocabStore) vs daachorse's
3–110 KB automaton; IREE's linear bucket scan degrades with set size (llama-256: 10 ns/byte;
2056 specials: 363 ns/byte). See the crate for the full table.

## What makes it fast

| addition | replaces | measured gain |
|---|---|---|
| NEON-SIMD pre-tokenizer | oniguruma regex | split **24 → 214 MB/s** |
| MPHF VocabStore (bytes→id) | fwd+rev HashMaps + daachorse | ~free lookup, **5.8× less** matcher RAM |
| allocation-free hybrid merge | heap-only / O(n²) | merge **11.0 → 6.9 ns/B**, 0 allocs |
| thread-local pretoken cache | HF shared-`RwLock` + per-miss `String` | **+13–17%**, 0 locks, 0 allocs |
| multi-turn prefix cache | full re-encode each turn | **56×** |

## Caveats (read before quoting)

- **Research prototype**, not production. Byte-exact vs HF main on the **12 GPT-2-byte-level models**
  tested (llama-3.x, qwen-2.5/3/qwq, deepseek-r1, smollm3, olmo2, phi-4). SentencePiece-style models
  (gemma, mistral, phi-3.5, …) load but need the Metaspace pre-tokenizer + Replace normalizer — the
  machinery that already lives in `tokenizers`. That's the integration path, not a rewrite.
- Numbers are Apple Silicon; only the **POC↔fastokens** comparison is same-machine. The HF and Crusoe
  ratios are each from their own setup.
- The pretoken cache helps 1–4 threads and is neutral at 8 (memory-bandwidth bound) — on by default.
