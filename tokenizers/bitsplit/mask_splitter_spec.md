# bitsplit — mask splitter spec

A Rust library for **pre-tokenizer splitting as a bitstream program**. Not a general regex engine:
it covers exactly what `tokenizers` needs — the GPT-family regexes, the class-run family, and
literal delimiters — and it does so with one shared operator set so a new grammar is ~150 lines of
boolean algebra rather than a new hand-rolled FSM.

Lineage: Parabix (Cameron et al., *Boosting the efficiency of text processing on commodity
processors*) for the operator set and carry discipline; *Interleaved Bitstream Execution for
Multi-Pattern Regex Matching on GPUs* (MICRO'25, doi 10.1145/3725843.3756052) for the execution
model — fuse every instruction into ONE block-wise loop instead of one pass per instruction.

Status: proven in `scratch/bitsplit` for three grammars, all byte-exact against their
`atomsplit::fsm` counterparts.

---

## 1. Why it wins

An FSM decides one token per unpredictable branch, so its cost tracks bytes-per-token. A bitstream
program decides 64 bytes per register op, branchlessly, so its cost is flat.

```
                 ns/B (english, same run, aarch64)
  fsm_deepseek   ████████████████████████████████████████████████  2.93   4.7 B/tok
  fsm_cl100k     ██████████████████████████████                    1.86
  fsm_byte_level ████████████████████████████                      1.71
  bitsplit ds    █████████                                         0.58
  bitsplit cl100k████████                                          0.50
  bitsplit bl    ██████                                            0.41
```

The FSM spread across grammars is 71% (1.71 → 2.93); bitsplit's is 20%. **Flat cost across grammars
is the design invariant** — it means the per-grammar layer is thin, which is what makes the library
worth having.

---

## 2. Pipeline

```
  text  ─────────────────────────────────────────────────────────────┐
    │                                                                │
    ▼                                                                │
  classify (atomsplit)          one Atom tag per byte                │
    │                                                                │
    ▼                                                                ▼
  ┌─────────────────────── per 64-byte block, fused ────────────────────────┐
  │                                                                          │
  │  tags ──► LUT ──► dense 3-bit code ──► fill conts ──► 3 bit-planes       │
  │                                                        │                 │
  │                                            decode ─────┼──► class        │
  │                                                        │    streams      │
  │                                                        ▼                 │
  │                              grammar algebra (Parabix operators)         │
  │                                                        │                 │
  │                                                        ▼                 │
  │                                              starts │ keep  (u64 each)   │
  └──────────────────────────────────────────────────────┬───────────────────┘
                                                         │  ~n/8 bytes
                                                         ▼
                                                   emit walk ──► spans / sink
```

Everything inside the box lives in registers. The **only** intermediate that reaches memory is the
`starts` bitmap. That is the whole point of interleaved execution: the sequential model (one loop
per bitstream instruction, each materialising a full-length stream) is what the MICRO'25 paper
beats, and it is what we must not accidentally rebuild.

---

## 3. The universal intermediate

Every pre-tokenizer in `tokenizers` reduces to **one bit per byte: "a token starts here"**, plus
optionally "this byte is kept". That single contract is what makes the library abstract without
being a general engine.

```
  text     H  e  l  l  o  ,     w  o  r  l  d
  starts   1  .  .  .  .  1  1  .  .  .  .  .
  keep     1  1  1  1  1  1  .  1  1  1  1  1     (Removed behaviour drops the space)
           └──"Hello"──┘  ","  └──"world"──┘
```

Which collapses three families onto one emit:

| family | how `starts` is produced |
|---|---|
| class runs (WhitespaceSplit, Punctuation, Digits, Whitespace, Bert) | `c & !(c<<1)` per class; DROP/ISOLATE/KEEP_A are 3 bit ops |
| regex grammars (gpt2, cl100k, o200k, tekken, deepseek) | the algebra of §6 |
| literals (Metaspace `▁`, CharDelimiterSplit) | shifted-AND chain of byte compares |

And `SplitDelimiterBehavior` stops being per-grammar code — it is a post-pass on the mask:

```
  match     .  .  1  1  1  .  .        (a matched delimiter run)
  Isolated  .  .  1  .  .  1  .        starts at both edges
  Removed   .  .  1  .  .  1  .        + keep &= !match
  MergedWithPrevious                   drop the leading edge bit
  MergedWithNext                       drop the trailing edge bit
  Contiguous                           merge adjacent same-kind runs
```

---

## 4. Operator set

Parabix names, because they are the literature standard. Every operator takes a `&mut Carry` so
cross-block state is explicit and never forgotten.

| operator | meaning | cost |
|---|---|---|
| `advance(m, n)` | move markers forward n bytes | 1–2 ops |
| `advance_char(m, cont)` | move markers forward one **char** (§5) | ~5 ops |
| `scan_thru(m, c)` | move each marker past the run of `c` it sits in | 1 add |
| `match_star(m, c)` | Kleene closure over a class | 1 add |
| `span_upto(m, e)` | set every bit from marker to end marker | 1 sub |
| `fill_to_last(m, c)` | in each `c`-run, fill from run start through the LAST marker | 2 ops |
| `to_lead(x, cont)` | move each bit back to its char's lead byte | ≤3 steps |
| `run_start(c)` | `c & !(c<<1)` | 2 ops |

Do all of these in `u128` and read bit 64 as the carry-out. That one habit removes an entire class
of bug: a run reaching bit 63 puts its landing bit at 64 instead of vanishing, and `e - m` still
yields the correct in-block span.

```
  fill_to_last, worked (finds "start after the LAST newline in a ws run"):

    run of c   ┌───────────────────────┐
    c (ws)     1  1  1  1  1  1  1  1  .  .
    m  (nl)    .  1  .  .  1  .  .  .  .  .
    scan_thru  .  .  .  .  .  .  .  .  1  .     ← lands past the run
    (e-m)|m    1  1  1  1  1  .  .  .  .  .     ← run start .. last marker
                             ▲
                             the surviving "after last newline" bit is here+1
```

`(e - m) | m` rather than `e - m`: the latter only spans from the *first* marker when a run holds
several. This is the single least obvious identity in the whole library — write a test for it.

---

## 5. The transpose (bytes → bitstreams)

The measured cost centre: **45–73% of total before optimisation, 36% after.** Optimise it before
touching any grammar.

```
  tags     [Lt][Lt][Pu][Sp][Lt][Lt][Lt][Ap][Lt] ...        16 lanes at a time
              │
              │  vqtbl4q  (64-entry table, indexed by the RAW tag —
              │            refinements 0x10/0x20/0x16/0x26 index straight in)
              ▼
  code      0   0   2   4   0   0   0   6   0   ...        dense 3-bit code
              │
              │  fill continuations from the left: 2 steps, not 3.
              │  shift-1 makes every lane right at distance 1, so shifting
              │  THAT by 2 covers distances 2 and 3 — the most a 4-byte char needs.
              ▼
  code'     0   0   2   4   0   0   0   6   0   ...        every byte carries its CHAR's code
              │
              │  3 × (vtst + vand POW + 3 × vpaddq_u8)   ← the 64×8 transpose
              ▼
  p0 p1 p2   three u64 bit-planes of the code, for 64 bytes
              │
              │  decode: ~14 boolean ops
              ▼
  l  n  other  nl  sp  ws  apo        class streams
```

**Filled streams are the load-bearing decision.** Because every byte of a multi-byte char carries
its char's code, "previous char's class" is a plain `<< 1` and no rule anywhere does char-width
arithmetic. Give that up and every rule grows an `advance_char`.

Three cost rules learned the hard way:

1. **Reduce plane COUNT, not per-plane cost.** ~13 ops/plane on NEON is the floor; a dense 3-bit
   code needing 3 planes beat one-hot class bits needing 7 (0.33 → 0.22 ns/B).
2. **Derive, don't extract.** `ascii = lead & last_byte` (a char that is its own first and last byte
   is single-byte, i.e. ASCII) and `ascii_alpha = lm & ascii` (ASCII has no marks). Two `u64` ops
   replaced two SIMD extractions.
3. **Gate rare masks per block.** The CJK range test is behind one `vmaxvq`, so Latin text pays ~8
   ops for it instead of ~70.

Multi-byte-char predicates (the CJK range) belong in **vector space on `vext`-aligned b1/b2 → one
mask**, not as N separate byte-predicate bitstreams recombined with shifts.

---

## 6. A grammar, end to end

Worked on `"Hi, don't"` under GPT-2 byte-level. Classes: `l` letter, `o` other, `s` space, `a`
apostrophe (a sub-code of other, so it can be flagged).

```
  index      0   1   2   3   4   5   6   7   8
  text       H   i   ,   ␣   d   o   n   '   t
  class      l   l   o   s   l   l   l   a   l

  l          1   1   .   .   1   1   1   .   1
  other      .   .   1   .   .   .   .   1   .
  ws         .   .   .   1   .   .   .   .   .
  sp         .   .   .   1   .   .   .   .   .

  l<<1       .   1   1   .   .   1   1   1   .
  sp<<1      .   .   .   .   1   .   .   .   .
  other<<1   .   .   .   1   .   .   .   .   1

  l_start     = l & !(l<<1) & !(sp<<1)        1   .   .   .   .   .   .   .   1
  o_start     = other & !(other<<1) & !(sp<<1)  .   .   1   .   .   .   .   1   .
  ws_start    = ws & !(ws<<1)                 .   .   .   1   .   .   .   .   .
  steal       = ws & lb & !(ws>>1) & !eof     .   .   .   1   .   .   .   .   .
                                              ─────────────────────────────────
  starts                                      1   .   1   1   .   .   .   1   1
  flag = starts & apo                         .   .   .   .   .   .   .   1   .
```

Emit walk, with the scalar escape firing at bit 7:

```
  [0,2) "Hi"     [2,3) ","     [3,7) " don"     ⟨escape⟩ [7,9) "'t"   ← bit 8 skipped
```

`" don"` shows `steal` doing its job: `\s+(?!\S)` hands the run's last whitespace char to whatever
follows, which is exactly the ` ?` prefix of the next alternative. One rule covers both.

### Per-grammar checklist

1. Dense code table `[u8; 64]`, `tag → code`, cont = 7. Merge every class the grammar treats
   identically (deepseek: Letter+Mark share a code; gpt2: Mark is "other").
2. `decode(p0,p1,p2,valid) -> Cls`. Mask with `valid` **only** the class whose code is 0 — past the
   block end every plane reads 0.
3. Run starts, one line each.
4. Prefix / steal rules.
5. Bounded repetition, span clearing, absorption.
6. Carries + the edge peek.

---

## 7. Escape hatches

Not everything belongs in bit algebra. **Flag the bits, resolve them scalar-ly in the emit walk.**

Contractions (`'s 't 're 've 'm 'll 'd`) are variable-length, case-optional, and outrank every other
alternative. In bit algebra that is miserable; as an escape it is 20 lines. `flag = starts & apo`
costs one word-test per block, so ordinary text pays nothing.

Two bugs this cost, both worth a regression test:

- The escape must not re-consume the start bit **at its own end** — the algebra usually has one
  there (the letter run resumes) and consuming it emits an empty span.
- Contractions **chain**: `'re've`, `y'all'd've`. Loop until a match fails rather than handing
  control back to the algebra, whose start bit you are about to skip.

Same shape applies to any rare, awkward, variable-length rule.

---

## 8. Rules and pitfalls

**Prefer backward formulations.** State a rule as "my predecessor did X", never "my successor is
X". Forward needs `advance_char`, which silently drops the marker when the two chars straddle a
block edge — every cl100k failure was this. Backward needs only a shift carry.

```
  ✗ forward   o_prefix = o_start & next_is_letter ;  suppress advance_char(o_prefix)
                                                     └─ marker leaves the block, lost

  ✓ backward  osf = smear(o_start over its char's bytes)
              l_start &= !(osf << 1)               └─ one carry bit, always exact
```

**Carries at a block boundary.** One bit per stream, plus one per open scan:

```
        block b-1                    │                    block b
  ... 61  62  63                     │   0   1   2 ...
       ▲                             │   ▲
       └─ last byte's code ──────────┼───┘  seeds the fill and every `p1`
          cont word ─────────────────┼───►  to_lead underflow patch
          open scans (aa/nl runs) ───┼───►  carry-in at bit 0
                                     │
       peek 1 char forward ◄─────────┼───   makes every `n1` exact at the edge
```

Other pitfalls, each caught by the fuzzer:

- Mask shifted-in zeros at u64 edges **before** inverting, or runs truncate silently.
- The first span needs special dispatch (`starts |= 1` on block 0).
- Cont-resolution needs prev-chunk context (`vext`), so starts land on leads.
- Bounded repetition needs a single-byte fast path — three `advance_char`s collapse to one `<< 3`
  against `n & (n<<1) & (n<<2)`. Without it, dense digits run **0.58×** the FSM.
- One backward-in-time rule survives (`\s*[\r\n]+`'s "after the LAST newline"): patch the
  already-written bitmap. The patch must NOT fire on a bit that a punct tail's `[\r\n]*` already
  made a run start — the absorption cut the whitespace run, so a later newline cannot reach past it.

---

## 9. Testing

Non-negotiable, because every one of these caught a real bug:

1. Byte-exactness vs the reference FSM on every input, for **every** grammar.
2. Exhaustive 3-char and 4-char sweeps over `[space, \n, a, 1, !, \0, 世, \u{a0}, \t, ']`.
3. Structured fuzz (~200k cases) over a pool hitting every arm: CJK, marks, ZWJ, control, multi-byte
   whitespace, Other_Alphabetic symbols, contractions, punct+alpha.
4. **Every corpus re-sliced at 70 offsets.** This is what finds carry bugs; whole-file runs do not.

## 12. Layout

```
  src/
    lib.rs        Blk, primitives, builders, emit, emit_contr
    simd.rs       NEON transpose  (x86 sibling slots in beside it)
    ops.rs        the Parabix operator set + Carry
    grammar/
      byte_level.rs   cl100k.rs   o200k.rs   deepseek.rs   class_runs.rs   literal.rs
  bin/
    verify.rs     byte-exactness oracle vs atomsplit FSMs
    bench.rs      per-grammar throughput
```

Each grammar is independently testable against its reference FSM, which is what keeps the shared
layer honest.

---

## 13. Open

- **Does the library own classification?** Currently depends on `atomsplit::classify`. Fusing buys
  ~0.07 ns/B on English but up to ~0.3 on CJK, where classify (0.57) now exceeds the split (0.24).
  Design the block builder so classify can be fused later without changing the operator API.
- **x86 builder.** The scalar path covers it correctly but slowly; the `vpaddq` reduction has a
  natural `movemask` equivalent.
- **o200k / tekken.** Case-refined letter runs; the case refinement is already in the Atom tags
  (`0x10`/`0x20`), so it is a code-table change, not new machinery.
