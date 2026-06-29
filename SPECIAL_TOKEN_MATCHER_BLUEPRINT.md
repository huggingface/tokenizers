# Special-token matcher — implementation blueprint

A fast, low-memory replacement for daachorse/Aho-Corasick when extracting added/special
tokens from text. Verified in scratch across 22 real tokenizers and 6 workload types.

**Core idea:** don't build an automaton. Reuse the MPHF the tokenizer already has for
`token_to_id`, and match in three cheap layers, each making the next one rare:

1. **SIMD skip** — find the next *candidate* byte 16–32 B at a time (memchr / NEON classifier).
2. **1-deep trie reject** — route on the byte *after* the bucket's common prefix; non-matches die O(1).
3. **Length-hash confirm** — crop the input to each distinct token length and probe the MPHF.

On long-context prompts (the bulk of real traffic) this is ~0.03 ns/byte vs daachorse ~0.8
(10–40×), with the matcher adding only a few hundred bytes per bucket on top of the vocab store.

daachorse has to do a state move on each  bytes it travels vs the efficient and fast SIMD skip scan.

---

## 1. Why this beats Aho-Corasick

- **AC walks one automaton step per input byte** — a serial, data-dependent memory load every
  byte, over a large double-array (cache-hostile). Flat ~0.8–1.3 ns/byte regardless of content.
- **This design** SIMD-skips plain text and only works at candidate sites. Each probe is an
  **independent O(1) hash** → instruction-level parallelism; hot data is tiny → stays in L1.
- Memory: AC trie is 3 KB → 110 KB (grows with vocab). This adds **~138 B/bucket** because
  membership reuses the existing `VocabStore` MPHF (strings stored once, for `token_to_id`).

Where AC still wins: genuinely dense + length-diverse streams where you can't skip
(e.g. gemma where `\t`/`\n` are special-token starts → ~9% of code bytes are real candidates).
There it's ~par. Image-placeholder runs needn't be string-matched at all (processor injects ids). But that's THE case where you'd have <IMG><IMG>. 


---

## 2. Data structures

```
VocabStore (already exists, shared with token_to_id/id_to_token):
    ptr_hash MPHF over all added-token byte strings
    slab: Vec<u8>            // all token bytes concatenated, once
    entries:  [(off:u32, len:u16, id:u32)]   // in MPHF-slot order
    fn token_to_id(&[u8]) -> Option<u32> // hash -> slot -> byte-verify against slab

Buckets (the matcher metadata — the ONLY thing added; ~hundreds of bytes total):
    fb: [u8; 256]                  // first byte -> bucket index, 0xFF = none
    list: Vec<Bucket>
    single: Option<u8>             // Some(b) if exactly one bucket (memchr fast path)
    // SIMD candidate finding:
    firsts: Vec<u8>                // sorted distinct first bytes (for memchr2/3 dispatch)
    cand: [bool; 256]              // is this byte a bucket first byte?
    lo16: [u8; 16], hi16: [u8; 16] // NEON nibble-classifier tables
    nib_ok: bool                   // false if >8 distinct high nibbles (use scalar)

Bucket:
    prefix:  Box<[u8]>             // longest common prefix of every token in the bucket
    exact:   bool                 // a token equal to the prefix itself exists
    disc:    Box<[u16; 256]>      // byte AFTER prefix -> index into `sub` (0xFFFF = none)
    sub:     Vec<Box<[u16]>>      // per-disc-byte: distinct token lengths, DESC (longest first)
    // (lengths: Box<[u16]> for a 1-level variant; the 2-level `disc`/`sub` supersedes it)
```

Membership note: you MUST byte-verify (the slab compare in `token_to_id`) — a bare MPHF hash
can collide and would mis-id a random input crop. Bytes live once in the slab; no second copy.
Do NOT build a separate MPHF just for specials (ptr_hash has a ~fixed floor, ~110 KB for 22
keys) — reuse the existing vocab store.

---

## 3. Build

```
1. Collect added-token byte strings (special + non-special). Dedup. (They already live in VocabStore.)
2. Sort by (first_byte asc, len desc).
3. Group by first byte -> one bucket each:
     prefix = longest common prefix of the group (NOT capped; gemma's is 1 byte, llama's is "<|")
     pl = prefix.len()
     for each token t in group:
        if t.len() == pl: exact = true
        else:
           d = t[pl]                      // the discriminating byte
           push t.len() into sub[disc[d]] // create the sub-list on first sight
     sort each sub[..] length list DESC, dedup
     fb[first_byte] = this bucket index
4. firsts = sorted distinct first bytes; cand[b]=true for each.
5. Build nibble tables (Section 5). If >8 distinct high nibbles -> nib_ok=false.
6. single = (#buckets == 1) ? Some(firsts[0]) : None
```

Cost: O(total token bytes). Tables are tiny and built once.

---

## 4. Hot-path scan (v3 = the target implementation)

```
search = 0
while let Some(ms) = next_candidate(text, search):     // LAYER 1: SIMD skip
    rem = text[ms..]
    b   = list[fb[rem[0]]]
    pl  = b.prefix.len()
    if rem.len() < pl || rem[..pl] != b.prefix:         // common-prefix reject (cheap)
        search = ms + 1; continue
    step = 1
    if rem.len() > pl:                                   // LAYER 2: 1-deep trie route
        si = b.disc[rem[pl]]
        if si != NONE:
            for L in b.sub[si] (DESC):                   // LAYER 3: length-hash confirm
                if L <= rem.len() && vocab.token_to_id(rem[..L]).is_some():
                    emit(ms, L); step = L; break         // longest-first => first hit is longest
    if step == 1 && b.exact && vocab.token_to_id(rem[..pl]).is_some():
        emit(ms, pl); step = pl                          // token == prefix (shortest), checked last
    search = ms + step
```

`next_candidate` (Layer 1):
```
if single == Some(b):  memchr(b, ...)                    // 1 bucket: SIMD, the common case
else by firsts.len():
    2 => memchr2(b0,b1,...)                              // SIMD
    3 => memchr3(b0,b1,b2,...)                           // SIMD
    _ => find_in_set(...)                                // NEON nibble classifier, else scalar
```

**Correctness invariants** (assert hit-count == daachorse leftmost-longest in tests):
- Longest-match: lengths probed DESC, and the prefix token (len pl) is checked last (it's shortest).
- A disc-routed token has len > pl (it has a byte at position pl); the prefix token has len == pl.
  So routed-then-exact ordering preserves longest-first.
- `token_to_id` does the byte-equality verify, so no false positives.

### Prefix rejection: ours (full) vs IREE (capped) — confirmed
- **IREE caps the bucket prefix at 4 bytes.** In `special_tokens.c` the bucket holds `uint8_t
  prefix[4]` + `prefix_length` (1–4); it checks at most those bytes, then **linear-scans** the
  bucket's tokens (longest-first `memcmp`) to confirm.
- **Ours uses the full, uncapped common prefix** of the group (gemma = 1 byte, llama = `<|` = 2
  bytes; a bucket whose tokens all share more — e.g. `<|reserved_special_token_` — rejects on the
  whole thing). **This is the planned design**, verified correct against daachorse leftmost-longest.
- Why we can afford the longer prefix and IREE caps it: IREE *follows* the prefix with an O(bucket)
  linear scan, so a longer prefix check buys little and adds fixed cost — capping at 4 is the right
  trade for a scan-based confirm. We *don't* scan: a longer prefix kills more non-matching
  candidates **before** the O(distinct-lengths) MPHF probe, with no scan to amortize against — so a
  full-prefix reject is strictly better for our confirm step.

---

## 5. The NEON nibble classifier (Layer 1 for >3 buckets) — IMPLEMENT THIS

**Goal:** for 16 input bytes at once, which are in the candidate-first-byte set S? Branchless,
~8 instructions per 16 bytes, instead of a scalar load+index+branch per byte.

### In one line
For every input byte we ask one question: *is it the first byte of any bucket* — i.e. could a
special token start here? The scalar answer is one `cand[b]` load + branch **per byte**. The
classifier answers it for **16 bytes per instruction**, so the ~99.9% of bytes that start nothing
get skipped 16-wide. This is purely the candidate *finding* (Layer 1); once a candidate is found,
`fb[b]` routes to its bucket and Layers 2–3 confirm.

### What a nibble is (and why)
A **nibble** is half a byte — 4 bits, value `0..15`. Every byte splits into a *high* nibble
(top 4 bits) and a *low* nibble (bottom 4): `b = (high<<4) | low`. We use nibbles because `0..15`
is exactly a valid index into a **16-entry** SIMD lookup table (`TBL`/`PSHUFB`), so a single
instruction performs 16 table lookups at once — one per byte in a 128-bit vector.

Worked example, S = { `<`=0x3C, `[`=0x5B }:
- High-nibble rows used: 3 (for `<`) and 5 (for `[`) → `hi[3]=0b01`, `hi[5]=0b10`, all others 0.
- Low-nibble columns: `<` is (h=3,l=C) → `lo[0xC] |= 0b01`;  `[` is (h=5,l=B) → `lo[0xB] |= 0b10`.
- Test `<` (0x3C): `lo[0xC] & hi[0x3] = 0b01 & 0b01` → nonzero ✓
- Test `=` (0x3D): `lo[0xD] & hi[0x3] = 0 & 0b01` → 0 ✗   ·   Test `S` (0x53): `lo[0x3] & hi[0x5] = 0 & 0b10` → 0 ✗

So membership is two tiny lookups + one AND; vectorized, that's 16 answers in ~4 instructions.

### Idea: factor each byte into nibbles
`b = (high<<4) | low`, `high,low ∈ 0..15`. The 256 byte values form a 16×16 grid
(16 high-nibble rows × 16 low-nibble columns). Membership in S is a yes/no over that grid.
A nibble is a perfect `0..15` index for `TBL`/`PSHUFB`, which is why nibbles are used.

### Two tables + AND (bit-plane encoding)
Assign each high-nibble *row* that contains a member a unique bit `0..7`:
- `hi_tbl[h] = 1<<bit(h)` if row h has members, else 0
- `lo_tbl[l] = OR of 1<<bit(h) over every member (h,l) in S`

Then `matched(b) = (lo_tbl[b&0xF] & hi_tbl[b>>4]) != 0`.

Why exact: `hi_tbl[h]` is row h's single bit; `lo_tbl[l]` has that bit set iff `(h,l)∈S`;
AND nonzero iff both → iff `(h,l)∈S`. Empty row → `hi_tbl[h]=0` → never matches.

**Limit:** a u8 lane has 8 bits → set may span at most **8 distinct high nibbles**. Special-token
first bytes are a handful (e.g. `<`=0x3, `[`=0x5, `\t`/`\n`=0x0, `0xe2`=0xe → {0,3,5,e}), so it
fits. If exceeded: split S into ≤8-high-nibble subsets, classify each, OR results; or use range
compares (`vcgeq`/`vcleq`) for contiguous spans; else scalar fallback (`nib_ok=false`).

### Build the tables
```rust
fn nibble_tables(cand: &[bool; 256]) -> Option<([u8;16],[u8;16])> {
    let (mut lo, mut hi) = ([0u8;16],[0u8;16]);
    let (mut next, mut bit, mut has) = (0u32, [0u8;16], [false;16]);
    for h in 0..16 { if (0..16).any(|l| cand[(h<<4)|l]) {
        if next >= 8 { return None; }                 // >8 high nibbles -> caller uses scalar
        let b = 1u8 << next; next += 1; hi[h]=b; bit[h]=b; has[h]=true;
    }}
    for h in 0..16 { if has[h] { for l in 0..16 { if cand[(h<<4)|l] { lo[l] |= bit[h]; } } } }
    Some((lo,hi))
}
```

### Scan (aarch64; NEON is baseline, no runtime detection needed)
```rust
#[cfg(target_arch = "aarch64")]
unsafe fn find_in_set(hay:&[u8], lo:&[u8;16], hi:&[u8;16], cand:&[bool;256]) -> Option<usize> {
    use core::arch::aarch64::*;
    let lov = vld1q_u8(lo.as_ptr());
    let hiv = vld1q_u8(hi.as_ptr());
    let m0f = vdupq_n_u8(0x0f);
    let (n, mut i) = (hay.len(), 0);
    while i + 16 <= n {
        let v   = vld1q_u8(hay.as_ptr().add(i));
        let lon = vandq_u8(v, m0f);          // low nibbles
        let hin = vshrq_n_u8(v, 4);          // high nibbles (already 0..15)
        let lm  = vqtbl1q_u8(lov, lon);      // lo_tbl[low]   (TBL == PSHUFB; idx>=16 -> 0)
        let hm  = vqtbl1q_u8(hiv, hin);      // hi_tbl[high]
        let m   = vandq_u8(lm, hm);          // nonzero lane == match
        // NEON has no PMOVMSKB: emulate movemask, 4 bits per lane, via shrn-by-4
        let t   = vreinterpretq_u16_u8(vtstq_u8(m, m));
        let packed = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(t, 4)), 0);
        if packed != 0 { return Some(i + (packed.trailing_zeros() as usize >> 2)); }
        i += 16;
    }
    while i < n { if cand[hay[i] as usize] { return Some(i); } i += 1; }   // scalar tail
    None
}
```

Key NEON facts:
- `vqtbl1q_u8(table, idx)` = per-lane `out[i]=table[idx[i]]`, `idx>=16 -> 0`. Nibbles are 0..15.
- NEON lacks `PMOVMSKB`. The **`vshrn_n_u16(x,4)`** trick packs the 128-bit compare into a u64
  with **4 bits per lane**: `trailing_zeros()>>2` = first match index, `count_ones()>>2` = #matches.
  (Simpler alternative: `transmute(m)` to `[u8;16]` and scalar-scan — fine, slower when dense.)

### Cross-platform
- **x86**: `_mm_shuffle_epi8` (PSHUFB) + real `_mm_movemask_epi8`, gated by
  `is_x86_feature_detected!("ssse3")` (or AVX2 for 32-wide). Easier movemask than NEON.
- **Portable one-source**: `std::simd` `Simd::swizzle_dyn` gives TBL/PSHUFB on both (nightly).
- aarch64 NEON needs no feature detection (baseline); x86 does.

---

## 6. memchr dispatch (Layer 1 for 1–3 buckets)

Most tokenizers have 1–2 distinct first bytes. Use the `memchr` crate (already a dep), which is
SIMD-optimized:
- 1 bucket → `memchr(b0, hay)`
- 2 buckets → `memchr2(b0, b1, hay)`
- 3 buckets → `memchr3(b0, b1, b2, hay)`
- ≥4 → NEON classifier above.

This alone took 2-bucket models (mistral, olmo) from 0.40 → 0.04 ns/byte on long context —
the candidate *finding*, not the checking, was the multi-bucket bottleneck.

---

## 7. Measured results (Apple Silicon, real chat-template workloads, ns/byte; NEW=v3, OLD=daachorse)

| workload  | example | NEW | OLD | note |
|-----------|---------|-----|-----|------|
| thinking  | mistral | 0.04 | 1.17 | 33× — long reasoning trace |
| paste     | mistral | 0.07 | 1.21 | 18× — 16 KB real C source |
| thinking  | gemma3  | 0.13 | 2.15 | 16× — NEON classifier, 6415 specials |
| conv      | llama   | 0.66 | 2.34 | 3.6× |
| paste     | gemma3  | 1.72 | 1.69 | ~par — `\t`/`\n` are real specials, can't skip |
| dense     | mistral | 3.70 | 3.02 | OLD wins — see §8 |

Also: token_to_id ~2× vs HashMap, id_to_token ~4× vs HashMap<u32,_>, total memory ~5.8× less.

---

## 8. Edge cases / TODO

- **Dense regions** (a candidate every ~2 bytes): `memchr`'s per-call SIMD setup costs more than a
  scalar peek. Detect a small last-gap and fall back to a tight scalar inner loop there. (Cheap;
  fixes the `dense` regression. Not yet implemented.)
- **>8 distinct high nibbles**: `nib_ok=false` → scalar `position`. Rare for special tokens.
- **Streaming / chunk boundaries**: a special may straddle a buffer edge — keep an
  `encode_state` (match_position + partial token index) and re-scan the bucket on the next chunk
  (see IREE `special_tokens.c` continuation path). Out of scope for the one-shot version.
- **Flags** (lstrip/rstrip/single_word): apply after a content match, before accepting (orthogonal
  to this matcher).
- **Their `extract_and_normalize` bug**: the existing multi-bucket path skips the common-prefix
  reject (goes straight to `match_bytes`); also `VocabStore::match_bytes` scans `entries[start..end]`
  in MPHF-slot order while bucket ranges assume sorted order — mismatch to fix.

---

## 9. References (for the SIMD parts)

- Wojciech Muła — "SIMD-ized check which bytes are in a set": `0x80.pl/articles/simd-byte-lookup.html` (canonical).
- simdjson — Langdale & Lemire, "Parsing Gigabytes of JSON per Second" + source `src/arm64/` (`simd8::lookup_16`, `classify`).
- Danila Kutenin — "Porting x86 vector bitmask optimizations to Arm NEON" (`danlark.org`) — the shrn-by-4 movemask.
- Daniel Lemire's blog (`lemire.me`); the `memchr` crate source (aarch64 paths, memchr2/3).
- ARM NEON Intrinsics Reference (`developer.arm.com`): `vqtbl1q_u8`, `vtstq_u8`, `vshrn_n_u16`, `vshrq_n_u8`.
- IREE `runtime/src/iree/tokenizer/special_tokens.c` — the original two-level bucket + prefix design.
