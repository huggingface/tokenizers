# For the blogpost — design decisions (bookkeeping)


## Candidate finding (scanning the input for special-token first bytes)
- `memchr` — single distinct first byte (1 bucket).
- `memchr2` / `memchr3` — 2–3 first bytes. **Tried, rejected**: IDK why but its slower than nibble.
- **NEON nibble classifier** (Muła / simdjson `shrn`-movemask, 4 bits/lane) for ≥2 buckets. 
- Final dispatch: **1 bucket → memchr, ≥2 → nibble** (dropped memchr2/3 entirely).
- Scalar fallback on non-aarch64; x86 AVX2/SSE2 (`pshufb`) port = TODO.
- Finding: SIMD candidate-finding 5–8× faster than scalar on sparse text; nibble *scan* throughput is density-invariant (popcount per 16-byte window).

## Scan iteration strategy
- **Restart (v1)**: re-enter the SIMD scan from `search+1` after every false candidate → reloads lo/hi registers + re-aligns the 16-byte window. Degrades hard with density (per-candidate reload).
- **Mask iteration (v2)**: classify each 16-byte window once, pop *all* candidate lanes from the mask, reload window only when drained or after a real match. lo/hi loaded once per scan. **Chosen.** 1.04–1.67× over restart, kills the dense cliff.

## Per-candidate rejection (the dominant cost on dense input)
- Two-layer fast reject: `starts_with(prefix)` → disc table (byte after prefix → length-list id, `0xFFFF` = none) → MPHF hash probe (lengths longest-first).
- Prefix check v1: walk the `Box<[u8]>` prefix (heap deref + loop).
- Prefix check v2: **pack `prefix[0..min(len,8)]` into a `u64` + mask** → single `(chunk ^ word) & mask` compare. Native-endian (`from_ne_bytes`, no byte-swap on any arch). **Chosen.** 1.7–3.3× on top of mask.
- Mask + fast-reject combined: **4.3–4.6× over restart on dense input; beats daachorse through ~50% candidate density.**
- Key insight: false candidates dominate dense input → rejection must be nearly free; that's the highest-leverage change.

## Matcher data structure (the "longest match" lookup)
- **MPHF VocabStore** (ptr_hash, single byte slab, slot-ordered). **Chosen.** ~5.8× less memory than HF's two hashmaps + automaton.
- daachorse (Aho-Corasick automaton): density-flat, but walks failure links on a miss; only wins at pathological >80% candidate density.
- IREE-style linear scan (first-byte group + byte compare, no MPHF/disc): 5.5–35× slower than `match_bytes`.
- Bucket prefix = LCP capped at `shortest_len-1` (guarantees a discriminating byte; sidesteps the token==prefix case). IREE caps its prefix at 4 bytes.

## Allocation
- `match_bytes` loops internally, single pass, no per-call allocation.
- Allocation-free merge (pre-allocated scratch, reserve-for-worst-case).
- Long added tokens (e.g. Llama-3 `<|reserved_special_token_N|>`, 25-byte shared prefix): `u64` covers the first 8 bytes; the >8 tail `starts_with` runs only when the first 8 match (rare). Long tokens cost ≈ short ones on the common path; even the worst case (full prefix + valid disc, dies at the hash) still beats daachorse ~1.75×.

## Normalized matching (planned, not done yet)
- Two-pass: non-normalized tokens on the raw bytes (pass 1, done) → normalized tokens on the `None` gaps (pass 2).
- Normalize only the gaps, reuse one scratch buffer, push straight into `splits`, fast-out when `normalized_vocab` is empty (IREE-lean, minimal alloc).
- Matched special tokens are *never* normalized (the serving hot path stays allocation-free).
