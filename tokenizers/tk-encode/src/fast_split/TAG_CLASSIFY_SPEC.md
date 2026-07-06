# Tag-classify pretokenization — design spec

**Principle (simdjson move):** compute a per-char **tag** stream in *one* SIMD pass, then every
pretokenizer is a cheap consumer over that stream. Two orthogonal tag alphabets (**atoms** for
category-based pretokenizers, **scripts** for `UnicodeScripts`) are produced by the *same* generic
range-classify engine — only the tables differ. 0-dep at runtime (tables bake to `const`;
`unicode_categories` is a dev-only *generator* dep, like `bitmap_gen`).

```
                       ┌── classify<Lanes, Table> ──► tag[]  (one SIMD pass, char-start aligned)
 text (UTF-8 bytes) ───┤
                       └── char-start bitplane      (for multibyte / char-count)
                                     │
   tag = ATOM (u4)                   │                    tag = SCRIPT (u8)
     │                               │                          │
     ├ remap(vqtbl1) → class-view ───┤                          └ FSM: split on script change
     │                               │                              (Common/Any/Inherited transparent)
     ├ RunSplit FSM (SIMD boundary + extract)   ← WhitespaceSplit, Whitespace, Punctuation, Digits, Bert
     └ Cl100k/ByteLevel FSM (scalar, segmented) ← cl100k, GPT-2
```

---

## 1. The atom alphabet (12 tags, fits `u4`)

Derived by enumerating all 1.1M codepoints against the exact predicates in the pretokenizer code
(`scratchpad/atomgen`). 10 base atoms + the two ASCII specials promoted (Space, Apostrophe).

| # | atom | definition | example |
|---|---|---|---|
| 0 | `Letter` | `\p{L}` | `a 中 é` |
| 1 | `NumWord` | `Nd ∪ Nl` (numeric **and** `\w`) | `0-9 Ⅷ ٠` |
| 2 | `NumOther` | `No` (numeric, not `\w`) | `½ ² ¼` |
| 3 | `Newline` | `\r \n` | |
| 4 | `Space` | `0x20` only | ` ` |
| 5 | `WsOther` | `\s ∖ {\r\n, 0x20}` | `\t` NBSP U+2028 |
| 6 | `Mark` | `\p{M} ∪ {ZWJ,ZWNJ} ∪ (Other_Alphabetic∖L)` | combining marks |
| 7 | `Connector` | `\p{Pc}` | `_ ‿` |
| 8 | `Punct` | `(\p{P}∖Pc) ∪ ASCII-sym`, minus `0x27` | `! . , + = < $` |
| 9 | `Apostrophe` | `0x27` only | `'` |
| 10 | `SymOther` | non-ASCII `\p{S}` ∪ control ∪ unassigned | `× ©` U+0000 |
| 11 | `NumericOther` | `is_numeric ∖ \p{N}` | numeric-typed non-numbers |

`Cont` (continuation byte) is the 13th value or, equivalently, tag `0xFF` + `char_start=0`; it's
transparent to every FSM. Base predicates that forced the split (proven, not guessed):
`is_alphabetic ≠ \p{L}` (27043 cps), `is_numeric ≠ \p{N}` (478 cps → atom 11), ASCII-vs-Unicode
symbol asymmetry (ASCII sym → 8, non-ASCII sym → 10).

## 2. The generic classify engine (composable lanes)

`classify` walks 16-byte chunks. Per chunk it runs **lanes**, each gated by a cheap
`vmaxvq` range test so ASCII-only chunks skip the multibyte lanes entirely. Every lane maps its
byte-region to a tag; the *structure* is shared across tag alphabets, only the per-lane **table**
changes. A `Classifier` = an ordered set of lanes + their tables.

| lane | byte range | mechanism | shared? |
|---|---|---|---|
| **ASCII** | `< 0x80` | `LUT128[byte] → tag` (nibble-shuffle / range) | structure shared; table per-alphabet |
| **2-byte** | `C2–DF` | codepoint bitmap (`vqtbl` `LETTER2`…) *or* range | per-alphabet |
| **CJK** | `E3–EC` | lead-byte range → tag (atoms: `Letter`; scripts: `Han`) | structure shared |
| **3-byte-other** | `E0–E2, ED–EF` | **SIMD range-search** (Thai/Devanagari; the must-fix) | structure shared |
| **4-byte** | `F0+` | SIMD range-search (rare) | structure shared |
| **cont** | `80–BF` | tag = `Cont`, `char_start=0` | shared |

**SIMD range-search kernel** (used by the 3-byte/4-byte lanes, and the whole script classifier):
a sorted threshold list classifies branchlessly by *counting thresholds cleared* —
`range_id = Σᵢ (cp ≥ tᵢ)` via broadcast `vcgeq` + accumulate, then `tag = TAG_LUT[range_id]`
(a `vqtbl1`). No binary search, no data-dependent branches, 16 lanes at a time. This is why the
3-byte-non-CJK marks/letters gap has **no scalar slow path**.

Output: `tag[]` (one byte per input byte; continuation bytes = `Cont`) + a `char_start` bitplane.

## 3. Composition: atoms and scripts are the same engine

```
AtomClassifier   = { ASCII: ATOM_ASCII_LUT, 2byte: ATOM_BITMAPS, CJK: →Letter,
                     3byte: ATOM_GC_RANGES, 4byte: ATOM_GC_RANGES }
ScriptClassifier = { ASCII: SCRIPT_ASCII_LUT (mostly Common/Latin), 2byte: SCRIPT_RANGES,
                     CJK: →Han,  3byte: SCRIPT_RANGES, 4byte: SCRIPT_RANGES }
```

Same lane code, same gating, same SIMD range kernel — swap the tables. `UnicodeScripts`'
`fixed_script` (Hira/Kata→Han, space→Any) is a post-remap, the same slot atoms use. So
`UnicodeScripts` reuses the entire classify + FSM framework; it is *not* a second system, it's a
second instantiation. (Script ⊥ atom at the *value* level — `'a'`/`'а'`/`'中'` are one atom, three
scripts — so they remain separate streams; you run whichever the configured pretokenizer needs.)

## 4. Pretokenizer interface

```
PreTokenizer = {
    classifier: &Classifier,     // Atoms (almost always) or Scripts
    remap:      Remap,           // tag → consumer class
    fsm:        Fsm,             // class-view → spans
}
```

- **`Remap`**: for `≤16` tags (atoms) it's a **16-byte `vqtbl1`** table — 16 chars remapped in one
  instruction. For scripts (`>16` tags) there is no small remap; the FSM compares script-ids directly.
- **`Fsm`** kinds (all local over the remapped tag view; the boundary stream is `tag ∈ mask`):
  - `Split { delim_mask, behavior }` — the HF delimiter split. A char is a *match* iff `tag ∈
    delim_mask`; matches are per-char, non-matches are runs; then `behavior ∈ {Removed, Isolated,
    Contiguous, MergedWithPrevious, MergedWithNext}` places boundaries / drops the match segments
    exactly as `SplitDelimiterBehavior` (normalizer.rs). Covers WhitespaceSplit (`WS`,Removed),
    Punctuation (`Punct∪Pc∪Apostrophe`,Isolated), Digits (`numeric`,Contiguous),
    Metaspace (`Space→▁`,MergedWithNext), CharDelimiterSplit (byte), Split-literal.
    → SIMD class-change → movemask → `extract`; `Contiguous`/`Removed`/`Isolated` are pure boundary
    masks (bitmask *wins* here), `MergedWith*` a one-lane shift.
  - `ClassRuns { drop_mask, isolate_mask }` — cut at *every* class change; drop `drop_mask` runs,
    isolate `isolate_mask` per-char. Covers Whitespace (drop `WS`, keep Word+Symbol runs) and Bert
    (drop `WS`, isolate `Punct`). This is the case that keeps *two* run types, so it isn't a single
    `Split`.
  - `Cl100k` / `ByteLevel` — the 7-rule (GPT-2) **scalar** FSM (segmented `{1,3}` / ws-tail — measured
    to beat the bitmask here). Peeks original bytes for the contraction suffix literals (ASCII).
  - `ScriptRun` — run-split on script change, transparent set `{Common, Inherited, Any}` sticks to the
    neighbouring run.

FSMs read the **tag stream** as primary input and may peek the **original bytes** for literal matches
that tags can't encode (cl100k contraction suffixes `'s`/`'re`, Metaspace marker, delimiter char).

## 5. Remap tables (atoms → consumer class), all `[u8;16]`

Atom index:  `0 Letter · 1 NumWord · 2 NumOther · 3 Newline · 4 Space · 5 WsOther · 6 Mark · 7 Connector · 8 Punct · 9 Apostrophe · 10 SymOther · 11 NumericOther`

```
                 0    1    2    3    4    5    6    7    8    9   10   11
cl100k        [  L,   N,   N,  NL,  SP,  WS,   O,   O,   O,  AP,   O,   O ]   // AP triggers rule 1; SP is rule-4 ` ?`
ByteLevel     [  L,   N,   N,  WS,  SP,  WS,   O,   O,   O,   O,   O,   O ]   // no \r\n split
Whitespace    [Wrd, Wrd, Sym,  ws,  ws,  ws, Wrd, Wrd, Sym, Sym, Sym, Sym ]
WhitespaceSpl [  ·,   ·,   ·,  WS,  WS,  WS,   ·,   ·,   ·,   ·,   ·,   · ]
Punctuation   [  ·,   ·,   ·,   ·,   ·,   ·,   ·,   P,   P,   P,   ·,   · ]
Digits        [  ·,   N,   N,   ·,   ·,   ·,   ·,   ·,   ·,   ·,   ·,   N ]
Bert          [  O,   O,   O,  WS,  WS,  WS,   O,   P,   P,   P,   O,   O ]   // ws-split + isolate punct
```

## 6. Outside the tag substrate

- `Split(regex)` — runtime pattern; **feature-gated**, rarely used. Escape hatch, not designed for.
- `CharDelimiterSplit` — arbitrary single byte; a byte compare, touches no tag.
- `FixedLength` — positional char count; rides the `char_start` bitplane only.
- Metaspace / cl100k-apostrophe — literal-byte matches; FSM peeks original bytes (§4).

## 7. Speed contract

- **classify** is the one hot pass; keep the ASCII/2-byte/CJK fast lanes, SIMD range for the rest.
  No scalar slow path (range kernel is branchless SIMD).
- **remap** is 1 `vqtbl1` / 16 chars.
- **RunSplit** pretokenizers extract via SIMD boundary bitmask (this is where the bitmask *wins* —
  no segmented rules).
- **cl100k/ByteLevel** stay scalar-FSM (segmented rules; measured winner vs onig 5–45×, byte-exact).
- byte-exactness gate: every FSM's output must equal its reference (regex / current impl) over a
  multi-script corpus. That single assert has caught every bug in this line of work.
