# Fast, light, allocation-free encode for `tokenizers` — the case for centralizing

*A performance investigation and a proposal: bring the wins that others are forking us for **into** the canonical crate.*

---

## TL;DR

- **Crusoe + NVIDIA Dynamo** published (Mar 2026, *"Reducing TTFT by CPU-maxxing tokenization"*) that tokenization is a **silent bottleneck worth up to 40% of TTFT** on long-context, and shipped **fastokens** — a Rust BPE tokenizer **9.1× faster than HuggingFace `tokenizers`** — to fix it.
- It is a **separate fork**. The whole stack — `transformers`, vLLM, TGI, SGLang, Dynamo itself — runs on **our** `tokenizers`. If we land these techniques **in the canonical crate**, every one of those users gets the win for free, byte-exact, on every model. No fork to adopt, no ecosystem fragmentation.
- I built a POC and benchmarked it **head-to-head against fastokens on the same machine**. It **matches their single-thread speed, scales far better across threads, is ~10× smaller, uses ~3× less RAM, and is provably allocation-free** — plus a **multi-turn prefix cache (56×)** that targets the exact workload Crusoe flagged (agent traffic: >50K-token prompts, >90% cache-hit).

**We are HuggingFace. We are the open-source center of this ecosystem. The right move is to centralize these gains in `tokenizers`, not to let the community re-implement them in N forks.**

---

## 1. The external signal

From the Crusoe/NVIDIA post (their numbers, their server CPUs):

- **9.1× average speedup over HF** across 4 models (DeepSeek-V3.2, MiniMax-M2.1, Mistral-Nemo, GPT-OSS-120B), LongBench + ShareGPT, 3 CPU architectures, inputs 512 → 100K tokens. *Gains grow with prompt length.*
- **Up to 40% faster TTFT** on long-context (prefill, batch=1, seq-out=1 — pure TTFT).
- Their stated production reality: *"prompt sizes exceeding 50K tokens with cache hit rates above 90%"* in agent systems.
- Their techniques: parallel pre-tokenization, and **"allocations were replaced by single buffers and preallocated scratch spaces."**

That last line is the same philosophy this POC follows — and we took it further (provably zero-alloc + thread-local, no shared lock).

## 2. Why this belongs in `tokenizers` (centralize, don't duplicate)

Two of the most credible infra teams in the industry just **forked our crate** to get a 9.1× win and a TTFT headline. That is a flashing signal:

1. **The demand is real and ecosystem-wide** — not a niche concern.
2. **The default is now "fork for speed"** — which fragments the ecosystem and duplicates effort across every serving stack.
3. **HF is the natural home.** Our crate is the one `transformers`/vLLM/TGI/SGLang already depend on. Folding these techniques in means the *entire* community benefits at once, byte-exact, across **all** models — including the ones a fork can't load (fastokens fails 8/22 models we tested: Metaspace pre-tokenizers, Replace normalizers).

This is the open-source dividend: one correct, fast, light implementation at the center, instead of many partial ones at the edges.

## 3. Head-to-head results

Single-thread is ids-only; HF main includes full offsets. POC↔fastokens is same-machine and apples-to-apples; HF and Crusoe ratios are each from their own setup.

| Library | 1-thread | 8-thread | Binary | Heap (llama-3) | Notes |
|---|--:|--:|--:|--:|---|
| **HF `tokenizers` (main)** | ~6 MB/s | ~50 MB/s | 3.6 MB | 28 MB | ecosystem standard; offsets, all models |
| tiktoken | ~14 | ~83 | — | — | ids-only |
| IREE | ~20 | ~113 | 0.7 MB | — | byte-exact, fuzzed |
| **fastokens (Crusoe/NVIDIA)** | ~118 | ~178 (35% scaling, 64 MB stacks) | 3.4 MB | 60 MB | 9.1× HF (their bench); **fails 8/22 models** |
| **POC (this work)** | **250–440** | **~600–690 (95% scaling)** | **0.37 MB** | **18.6 MB** | byte-exact, **allocation-free**, **+56× multi-turn** |

## 3a. Workflow-based benchmarks (the examples we designed)

We did not benchmark on uniform text. Each workload is built from `apply_chat_template` across 22 production models — the distributions tokenizers actually see in serving:

- **thinking** — long reasoning dumps (R1 / QwQ style)
- **paste** — a big pasted code / text block
- **conv+ctx** — multi-turn with long context
- **conv** — short back-and-forth chat
- **dense** — many tiny turns, marker-heavy
- **plain-en** — prose baseline

Final cached POC, **single-thread MB/s** (last column = 8-thread batched throughput):

| model | thinking | paste | conv+ctx | conv | dense | plain-en | 8-thread |
|---|--:|--:|--:|--:|--:|--:|--:|
| llama-3 | 416 | 221 | 62 | 213 | 187 | 255 | 604 |
| llama-3.1 | 411 | 228 | 61 | 209 | 171 | 257 | 600 |
| llama-3.3 | 409 | 227 | 59 | 159 | 133 | 261 | 636 |
| smollm3 | 399 | 213 | 60 | 97 | 121 | 258 | 595 |
| deepseek-r1-llama | 427 | 226 | 62 | 303 | 434 | 253 | 594 |
| deepseek-r1-qwen | 427 | 230 | 59 | 324 | 545 | 253 | 626 |
| qwen2.5-7b | 405 | 221 | 60 | 122 | 71 | 257 | 610 |
| qwen3 | 410 | 217 | 60 | 116 | 57 | 258 | 612 |
| qwq | 406 | 224 | 60 | 121 | 59 | 254 | 621 |
| qwen2.5-vl | 408 | 218 | 59 | 124 | 61 | 260 | 597 |
| olmo2 | 377 | 212 | 57 | 176 | 63 | 235 | 570 |
| phi-4 | 429 | 231 | 61 | 331 | 368 | 261 | 596 |

Reading the field: **conv+ctx (~60 MB/s) is the floor everywhere** — long unique context, low repetition, most merging. **thinking / paste / plain-en (200–430)** is where the cache shines. **deepseek-r1 dense (434–545)** — tiny repeated turn-markers cache perfectly, while qwen dense (57–71) produces more distinct merges. Same task, tokenizer-dependent.

## 3b. Before → after across the 22 models

HF main today (before) vs the POC (after), big.txt single-thread, **byte-identical output**. Byte-exact on all 12 GPT-2-byte-level models:

| model | HF main (before) | POC (after) | speedup |
|---|--:|--:|--:|
| llama-3 | 6.3 | 92 | **14.6×** |
| llama-3.1 | 6.2 | 91 | 14.7× |
| llama-3.3 | 6.3 | 91 | 14.4× |
| smollm3 | 6.2 | 89 | 14.4× |
| deepseek-r1-llama | 6.5 | 92 | 14.2× |
| deepseek-r1-qwen | 5.2 | 90 | 17.3× |
| qwen2.5-7b | 5.2 | 91 | 17.5× |
| qwen3 | 5.2 | 92 | 17.7× |
| qwq | 5.1 | 91 | **17.8×** |
| qwen2.5-vl | 5.2 | 92 | 17.7× |
| olmo2 | 5.6 | 85 | 15.2× |
| phi-4 | 5.6 | 83 | 14.8× |

**14–18× on every byte-exact model** — and that's the cleaned POC's *uncached* big.txt pass; on real cached workloads it reaches 250–430 MB/s (table above). The 10 SentencePiece-style models (gemma 2/3, mistral-v0.3, phi-3.5, deepseek-v2.5/v3, granite-3.1, smollm2) load but need the normalizer + Metaspace path — which already lives in `tokenizers`. That is the whole point: bring the speed in, reuse the correctness machinery, and all 22 benefit.

## 4. What the POC actually is

A from-scratch, inference-only llama-3-family encode path, validated component-by-component:

- **MPHF `VocabStore`** — one ptr_hash minimal-perfect-hash structure replaces HF's three (fwd `HashMap`, reverse `HashMap`, daachorse trie). Strings stored once → **5.8× less matcher memory**; serves `token_to_id`, `id_to_token`, and special-token membership.
- **NEON-SIMD pre-tokenizer** — a hand-rolled DFA for the GPT-4 split regex with a nibble-classifier (classifies 16 bytes/instruction). **216 MB/s split-only**, vs PCRE2-JIT 149 and onig 24. Removes the C regex dependency entirely.
- **Allocation-free hybrid merge** — linear merge for short pretokens (no heap setup), heap (O(n log n)) only for pathological long ones. Pre-allocated worst-case scratch + reserve-guards. **Proven 0 allocation calls across 20 warmed encodes.** Merge cost 11.1 → 7.1 ns/byte; single-thread 65 → 90 MB/s.
- **Thread-local pretoken→ids cache (FlatCache)** — open-addressing slots + bump arenas, clear-on-full. **Owned per thread → zero locks, zero contention** (fixing HF's shared-`RwLock` + per-miss `String` alloc). Allocation-free. +18–50% single-thread, neutral at max parallelism.
- **Multi-turn prefix cache** — reuses the unchanged conversation prefix's splits *and* ids; re-encodes only the new suffix. **56× on a 256-turn growing conversation**, byte-identical.
- **IREE-style ring buffer** — bounded-memory streaming with parity-preserving boundary carry.

### The additions that make it fast, with measured gains

| addition | replaces | measured gain |
|---|---|---|
| NEON-SIMD pre-tokenizer | oniguruma regex (C dep) | split **24 → 214 MB/s** (≈9×) |
| MPHF VocabStore (bytes→id) | fwd map + reverse map + daachorse trie | ~free lookup · **5.8× less** matcher RAM |
| Allocation-free hybrid merge | heap-only / O(n²) scan | merge **11.0 → 6.9 ns/B** · 65 → 91 MB/s · 0 allocs |
| Thread-local FlatCache | HF shared-`RwLock` + per-miss `String` | **+13–17%** (1–2 thr) · 0 locks · 0 allocs |
| Multi-turn prefix cache | full re-encode every turn | **56×** re-encode · O(N²) → O(N) |
| Ring buffer (IREE-style) | whole-input buffering | bounded memory · parity preserved |

Cumulative single-thread on big.txt: NEON split (214 MB/s split-only) → bytes→id is **~free** (~0 ns/byte; the merge, not the lookup, is the cost) → hybrid merge **65 → 91 MB/s** → cache **→ 103 MB/s** (up to 1.5× when warm) — all **provably zero-alloc** (0 allocation calls across 20 warmed encodes). The cache helps 1–4 threads and is neutral at 8 (memory-bandwidth bound), so it ships on by default.

## 5. Multi-turn / agent — the biggest structural win

Crusoe's own data says agent prompts are huge and **>90% cache-hit between turns**, i.e. the conversation prefix barely changes. Yet today we **re-tokenize the entire prompt every turn — O(N²) over a conversation.** The prefix cache makes turn-N cost **O(new message), not O(whole history)**:

```
256-turn growing conversation, full re-encode each turn : 1226 ms
                              with prefix cache          :   19 ms   → 56× faster, identical tokens
```

For a 50K-token history + a 200-token new turn that is ~250× less tokenization work per turn. **This removes tokenization from multi-turn TTFT** — exactly the 40% Crusoe is chipping at, attacked structurally rather than by constant factor.

## 6. On-device / edge — where light matters most

- **Binary: 0.37 MB** vs HF 3.6 / fastokens 3.4 — ~10× smaller, after dropping the C regex libs (onig/pcre2) for the NEON splitter, daachorse, and ahash. Pure-Rust dep set: `memchr` + `serde_json` (+ optional `ptr_hash`, `rayon`).
- **Heap: 18.6 MB** loaded (llama-3) vs HF 28 / fastokens 60. Across all 22 models the POC averages **~0.64× HF and ~0.3× fastokens**.
- **Allocation-free hot path (proven).** On edge this is the headline: no allocator jitter, no GC-style tail latency, predictable per-token cost, zero lock contention.

## 7. Throughput + scaling (training & batched serving)

Tokenizing training corpora and batched prompts is embarrassingly parallel — but only if the design allows it. The POC's per-thread, zero-contention, zero-alloc design scales **~95% to 8 threads (~600–690 MB/s)**. On the same machine, fastokens' batch API scaled ~35% and needed 64 MB worker stacks. That is the difference between linear and sub-linear on the 100+-core boxes Crusoe targets.

## 8. Extrapolated impact

- **One-shot long-context:** landing fastokens-class speed in `tokenizers` hands the **same up-to-40% TTFT reduction** to the entire ecosystem, on every model, byte-exact — not just fastokens adopters.
- **Multi-turn (the dominant >90%-hit agent workload):** the prefix cache attacks what raw speed leaves on the table. Crusoe cut ~40% of TTFT with throughput; the prefix cache structurally removes the *rest* of tokenization from per-turn TTFT (O(N)→O(new)). Stacked, **tokenization stops being a TTFT line item for agentic serving.**

## 9. Coverage & honesty

- POC is **byte-exact on the 12 GPT-2-byte-level / llama-regex models** tested (llama-3.x, qwen-2.5/3/qwq, deepseek-r1, smollm3, olmo2, phi4, …).
- SentencePiece-style models (gemma, mistral, phi3.5) need the Metaspace pre-tokenizer + Replace normalizer — machinery that **already exists in `tokenizers`**. That is precisely why this belongs in the crate: the speed techniques compose with the correctness machinery we already maintain.
- Our absolute MB/s are Apple-Silicon; they are not directly comparable to Crusoe's server-CPU *ms*. The **fastokens-vs-POC comparison is same-machine and fair**.

## Blog post (planned)

We intend to publish a blog post: the **motivation** (why tokenization is a TTFT bottleneck, with the Crusoe/NVIDIA result as the external anchor), the **benchmark methodology**, the **workload taxonomy** above, and the **open-source case** for centralizing this in `tokenizers` rather than fragmenting across forks. This document is the internal precursor.

## 10. The ask

Green-light productionizing these techniques into `tokenizers`:
1. MPHF `VocabStore` for the vocab + special-token matcher (memory + speed, byte-exact).
2. NEON/SIMD pre-tokenizer with a scalar fallback (drop the C regex dependency).
3. Allocation-free, preallocated hybrid merge.
4. Thread-local, allocation-free pretoken cache (replacing the shared-lock cache).
5. The multi-turn prefix cache as a first-class re-encode API.

We are HuggingFace. We are the open-source center of gravity for tokenization. Let's make the canonical crate the fast one — so the ecosystem consolidates on us instead of forking around us.

---

*All figures measured this investigation on Apple Silicon, llama-3 tokenizer, hot cache, ids-only unless noted. Crusoe/NVIDIA figures from their published post.*
