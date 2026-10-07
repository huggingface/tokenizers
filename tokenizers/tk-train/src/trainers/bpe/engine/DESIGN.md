# BPE training design

This document explains the engine's weighted BPE selection rules and how it
preserves their order using fixed corpus coordinates, parallel merge preparation,
and compressed occurrence lists. [README.md](README.md) covers the public entry points and test
commands.

## Weighted training semantics

Each word contributes its weight to every adjacent token pair. Repeated
occurrences contribute repeatedly: `abab` with weight 3 gives `(a, b)` a count of
6 and `(b, a)` a count of 3. Applying a selected rule takes nonoverlapping matches
from left to right.

A training call first tries `FirstActivationOnly`, also called fresh mode. It
selects the largest positive pair count, breaking ties by the smaller token-ID
pair. An existing fresh key can only lose count. Its cached queue count is
therefore an upper bound until selection repairs and certifies it. Once its
complete count falls below the selection floor, `max(min_frequency, 1)`, the key
can be retired permanently.

`AllowActiveReuse`, also called reuse mode, supports an already active ID becoming
a merge result again. It keeps a signed ledger and separate occurrence cohorts.
Its selection rules are specified in [Active ID reuse](#active-id-reuse).
Training ends when the vocabulary reaches its target size or the mode's queue
has no eligible winner.

### Token identity and initialization

[Vocabulary](vocabulary.rs) stores unique strings in insertion order; their indices
are token IDs. It inserts special tokens first, then visits retained alphabet
characters in codepoint order. A string already present keeps its ID, including
a character that is also a special token. Adding a token never renumbers an
assigned ID. A merged string also keeps its ID if it already exists. These
guarantees apply within one attempt. Each public training call initializes a new
vocabulary, and `train` replaces the public model.

Vocabulary membership and activation are separate. An ID becomes active when it
represents an initial corpus symbol or a selected merge result, and stays active
for the rest of that attempt. A reserved ID already has a string and an index but
has not yet been active. Activating it first uses fresh mode; selecting a result
whose ID is already active requires reuse mode.

With an unlimited alphabet, a bitmap collects observed characters and marks their
plain IDs active. Limited alphabets and nonempty affixes require a scan of the
borrowed input to mark the actual initial symbols active. That scan also allocates
new decorated IDs in input traversal order, before words are sorted by weight.

The limited-alphabet selector ranks characters by weighted frequency. It gives
configured initial-alphabet characters the maximum frequency, but a limit smaller
than that alphabet can still remove some of them. Equal-frequency truncation has
no complete codepoint tie break. The retained characters are then inserted in
codepoint order. Decorated IDs follow the caller's map traversal for `do_train`,
or the stored map or entries after `feed`. Map traversal and parallel feed entry
order are unspecified. Fixed pair tie breaks and worker parity for the same input
view do not guarantee identical initialization across arbitrary hash seeds.

Filtering and affixes use each character's position in the original UTF-8 word.
A continuing prefix applies when that position is not the first character; an
end suffix applies at the original last character. Removing a character does not
change those decisions for the remaining characters. A shared symbol scanner
implements this rule for initial edges and corpus construction. To form a merged
string, the engine joins the left string with the right string after removing a
matching continuing prefix from the right.

## Fixed coordinates and occurrence lists

[CorpusPlan](corpus/prepare.rs) borrows the input strings, orders words by
descending weight, measures retained symbols, and assigns coordinates. Each
retained symbol has one slot, and a separator ends each word. These coordinates
remain fixed throughout the attempt.

A live token stores its ID at its first and last slots. Its span is the number of
retained-symbol slots it covers. The span locates the next live token; the
preceding endpoint locates the previous token. A merge updates endpoints and
leaves the remaining slots of the word in place.

For example, consider four retained symbols with local coordinates:

```text
coordinate         0   1   2   3   4
initial token      A   B   C   D   separator
after AB -> X      X   X   C   D   separator
live starts        0       2   3
```

These are local coordinates for illustration; the corpus also has a leading
separator before the first word. `X` occupies the endpoints of a two-slot span.
Coordinate 1 is its last endpoint, not a second live token start. An old `(B, C)`
occurrence recorded at coordinate 1 is now stale, while the new `(X, C)` boundary
starts at coordinate 0. Occurrence lists can keep stale addresses. Before using
an address, the matcher checks current
endpoints. List length counts recorded addresses, including stale ones; the
weighted pair count is maintained separately.

Longer merges can leave unused interior slots holding old IDs or separator codes.
Live-token traversal follows spans from a word start, rather than treating each
nonseparator slot as a token. For a recorded occurrence, the matcher tests the
current left ID and the right ID reached by that left token's span.

Fresh mode normally obtains spans from the token-ID table. If active reuse gives
the same ID unequal occurrence spans, the corpus materializes a span plane before
the rewrite. Jobs then own complete whole-word regions so they can update those
occurrence spans safely. Immutable word boundaries retain word identity.

Reuse preparation always partitions at word boundaries. It scans complete words
when an ID has been reused or a birth length limit is configured; otherwise it
can visit the cohort's recorded positions within those regions. This preserves
intermediate boundaries needed for cohort accounting.

### Delayed corpus construction

`CorpusPlan` owns word geometry, resolved initial IDs, and checked resident-size
and numeric bounds. `WordMeasure` measures retained symbols once and derives the
UTF-8 seek anchors used by initial grouping. Bounded waves own left endpoints;
the last edge in a wave can read one symbol beyond its range.

Initial pair construction scans the borrowed word strings through the resolved
character-ID lookup tables. Checkpoints retain original UTF-8 coordinates,
including repeated slot coordinates after filtering, so bounded waves can seek
into long words. Record and collector admission use the resolver's initial-ID
bound and the observed alphabet.

Only after raw initial records have retired does the plan allocate mutable slots,
decoding the borrowed strings directly into the selected slot representation.
The plan's lookup tables and checkpoints are released after materialization.
This ordering avoids overlapping a corpus-sized symbol plane with raw initial
records and their sorting scratch.

When the initialized vocabulary already meets the target size, no mutable corpus
is needed. Planning still checks resident-size and signed-input bounds. If the
total weighted edge mass fits in `u64`, it proves that every pair count fits, and
training returns after planning. Otherwise initial grouping performs
the per-key checks before returning an empty merge list. Total mass above
`u64::MAX` is therefore not itself a plain-input error.

### The birth length rule

The internal gate accepts a newborn boundary only when its combined span is
strictly less than `max_token_length`. Spans count retained corpus slots. They
exclude affix text, filtered characters, and UTF-8 byte widths.

Initial pair counting bypasses this gate. Applying a selected pair has no separate
length check. Consequently, a limit below two can still admit an initial
two-symbol merge. The setting controls birth admission; it is not a uniform
maximum for output token strings.

For example, take only the plain word `abc` with weight 1, no special tokens,
`min_frequency = 1`, and a target vocabulary large enough to continue. The initial
tie selects `(a, b) -> ab`. With `max_token_length = 3`, the newborn `(ab, c)`
spans three slots and fails the strict gate, so training cannot form `abc` from
that boundary. With a limit of 4 it can. Even a limit of 1 permits the initial
`(a, b)` merge, because initial pairs bypass birth admission.

## Round ownership

The coordinator in [mod.rs](mod.rs) executes one round in this order:

```text
select -> prepare -> release candidates -> apply -> commit -> release events
```

Selection consumes candidates, resolves result IDs, and prepares span metadata.
At the end of selection, cached fresh prefixes return to their heaps so count
changes can invalidate them before refill.

Preparation reads one stable corpus snapshot. It returns write plans, neighbor
events, and any complete birth lists. All preparation tasks join before writes
begin. Candidate lists then have no readers and can be released before commit
allocates the next generation of positions.

`PreparedMerges::apply` consumes the plan and holds a mutable corpus borrow until
parallel writes join. Ordinary jobs own disjoint endpoint spans. Jobs using
occurrence spans own complete whole-word regions. Consuming the plan prevents a
second application of that value, and the mutable borrow excludes safe
concurrent corpus access. The type does not identify the corpus instance or
version: the coordinator must preserve the preparation snapshot through apply.

Commit updates each logical pair owner's state. An owner is a shard, while a
worker is the pool thread executing its task; work stealing can change the worker.
The reciprocal router gives the same owner as integer remainder. Routing changes
where a key is stored, never its priority.

Events own the position chains borrowed during commit. They stay alive until all
owners join. Only then can the coordinator release events and begin selection
again. [Failure behavior](#failure-behavior) specifies what happens if a phase
returns an error.

## Why batching preserves rule priority

[RuleBatch](batch.rs) selects an ordered prefix. It stops at the first incompatible
candidate and does not skip that candidate to accept a lower-priority rule.
Fresh batches contain at most 256 rules and are also capped by the remaining
vocabulary capacity. The cap may end a compatible prefix early.
Rules can share a left token or share a right token. Crossed endpoints are
forbidden: a later rule's left token cannot equal an earlier rule's right token,
and its right token cannot equal an earlier rule's left token.

For example, `(A, B) -> X` and `(C, D) -> Y` can be compatible. In the sequence
`A B C D`, the rules consume disjoint spans, and the final boundary is `(X, Y)`.
By contrast, `(A, B)` followed by `(B, C)` fails the crossed-endpoint check. In
`A B C`, applying the first rule consumes the second rule's left endpoint.
Shared heads, as in `(A, B)` and `(A, C)`, or shared tails, as in `(A, C)` and
`(D, C)`, do not by themselves cause overlapping matches.

Disjoint writes are only part of the argument. A batch must also preserve the
priority of later selected rules. The following argument assumes fresh mode,
distinct newly appended result IDs, and a certified priority for each selected
old key. Reserved IDs, AA rules, and active reuse use the separate paths below.
Under these assumptions:

1. The crossed-endpoint checks prevent an earlier accepted rule from consuming an
   endpoint required by a later accepted rule. The later rule's old pair count
   therefore stays unchanged.
2. Each newborn boundary descends from an old boundary `(L, R)`. Its occurrences
   are a subset of that old boundary's occurrences and keep their word weights.
   Its count cannot exceed the old boundary count.
3. The old boundary key cannot itself be an accepted rule: merging either side
   of that boundary would conflict with it in the crossed-endpoint check.
   Because selection takes an ordered prefix without skipping, the old key's
   priority cannot precede a later accepted rule. A key already below the floor
   also bounds its descendants below that floor.
4. For equal counts, replacing `L` or `R` with a newly appended ID increases the
   lexicographic pair key. The newborn boundary cannot move ahead of the accepted
   prefix on that tie.
5. Each appended result ID identifies one producer rule, so a given newborn key
   has one old boundary key as its witness. Different tasks can contribute to it,
   but their combined occurrences remain a subset of that same witness's
   occurrences. Reusing an active result ID would break this argument and is
   detected before the fresh batch is applied.

In the `A B C D` example, `(X, Y)` descends from `(B, C)`. The bound holds when
one or both sides of the boundary are replaced. Preparation recognizes selected
neighbors and emits the final newborn boundaries and checked removal mass in
sequential order.

The tie argument requires newly appended IDs. Reserved IDs can be smaller than
the IDs they replace, so reserved-ID rules run alone. AA overlaps and active
reuse also use single-rule rounds, as described next. [Batch tests](batch.rs),
[semantic traces](tests/semantic_parity.rs), and
[parallel ordering tests](tests/routing_and_publication.rs) cover these boundaries.

## Single rule cases

### AA overlap selection

For an AA rule, adjacent occurrence addresses overlap. A run `A A A A A` has four
AA starts, but left-to-right application selects starts 0 and 2. Splitting that
run into parallel chunks requires the chunks to agree on which starts to skip.
If the chunks own starts `[0, 1, 2]` and `[3]`, the first selects 0 and 2. The
second must skip 3 because the match at 2 already consumes that endpoint.

[AA parity](aa_parity.rs) summarizes each chunk's trailing run. Ordered summaries
determine whether the next chunk skips its first match. Workers then select
their own starts. The coordinator processes summaries rather than scanning every
occurrence, while the result remains exactly left to right.

### Reserved IDs

A vocabulary string can exist before it becomes active. Its first activation
keeps the reserved ID and uses fresh-mode counts. It runs alone because that ID
does not satisfy the increasing-ID part of the batch proof.

For a concrete counterexample, reserve special token `ad` as ID 0 and use plain
alphabet IDs `a=1`, `b=2`, `c=3`, `d=4`, and `e=5`. Give `ade` weight 6, `ad`
weight 4, and `bc` weight 6. The initial priorities are:

```text
(a, d) = (1, 4)   count 10
(b, c) = (2, 3)   count  6
(d, e) = (4, 5)   count  6
```

The first rule activates reserved ID 0. It creates `(ad, e) = (0, 5)` with count
6. That birth now wins the tie over `(b, c)`, even though its old boundary
`(d, e)` lost that tie. Accepting `(b, c)` in the same batch would therefore change
sequential rule order. This is first activation of a reserved ID; no active reuse
is required.

### Active ID reuse

An already active ID can become a merge result again. It can revive an old pair
key and represent occurrences with different spans. Fresh-mode pruning and
batching cannot supply the required history.

Fresh selection detects this collision before accepting the rule. It discards
the entire attempt and rebuilds from unchanged input under `AllowActiveReuse`,
retaining the chosen alphabet. Earlier accepted rules in that batch may already
have changed vocabulary or span metadata. Their output is not published. A
pruned fresh index cannot switch to the reuse ledger in place.

The suffix fixture in [identity tests](tests/identity_reuse.rs) provides a concrete
case. With word `baaba`, weight 1, and end suffix `a`, initial symbols are
`b a a b aa`. The suffix has already activated the string `aa`. The first merge
`(a, a) -> aa` therefore reuses its ID. The literal trace begins with `(a, a)` at
count 1 and then `(b, aa)` at count 2.

Reuse mode selects one birth cohort at a time. A cohort owns an independent
position list, while all cohorts for one pair key share a checked signed `i64`
ledger. Intermediate births remain observable. A candidate's priority snapshot
stores the ledger's bits as `u64`; selection repairs a stale snapshot only when
it reaches the global queue frontier. These snapshots lack fresh mode's
decreasing-count upper bound.

Selection accepts nonzero unsigned ledger bits at or above `min_frequency`.
A negative signed ledger therefore has a large unsigned priority. Publishing a
new cohort instead requires a positive signed ledger; positive cohorts remain
observable even below the selection floor. The negative-ledger fixture in
[PairIndex](pair_index.rs) protects this index-level distinction. It does not
establish that a public training input produces a negative ledger.

## Counts and publication

Three quantities serve different purposes. A position-list length counts stored
addresses, including stale ones. A fresh pair count measures the admitted
weighted occurrences; a reuse ledger records ordered signed changes shared by
its cohorts. A queue snapshot caches a priority and may need repair. Comparing
only position-list lengths cannot validate weighted counts or queue priority.

### Numeric domains

Engine pair counts, birth weights, and removal weights use checked `u64`
arithmetic. Reuse ledger updates use checked `i64` arithmetic. A nonempty prefix
or suffix, or a reuse attempt, requires both the maximum word weight and the
initial weighted edge mass to fit in `i64::MAX`. The affix check applies before
active reuse is detected. Empty affixes take the plain path.

Plain fresh input can have total weighted edge mass above `u64::MAX` if each
individual pair count fits in `u64`. Feed counting and limited-alphabet frequency
accumulation retain ordinary additions; they do not acquire the engine's checked
arithmetic contract.

The initial weighted edge mass is the sum of each word's weight times one less
than its retained-symbol count, with empty words contributing zero. The signed
check also covers word weights with no retained edges. Nonempty affixes enforce
these checks even when the initialized vocabulary needs no additional merges.

### Complete and partial producers

Ordinary candidates are packed whole into jobs when possible. Large lists are
split into ordered spatial ranges. A job that covers a fresh rule's entire
candidate list, and fits the node budget, can collect every occurrence of each
newborn key. It can then test the complete count against the floor and encode the
final positions during preparation.

This path returns `CompletedBirth { key, weight, positions }`. The weight includes
all contributions to the key, and the encoded list borrows the training arena.
Preparation retains removal events but emits no routed birth chains for that
record. Commit moves the encoded list into fresh pair state exactly once.
Preparation has no access to index state layout or ledger bits; commit owns state
construction and priority publication.

Split producers emit partial chains for owner reduction. A fragment below the
floor cannot be pruned independently, because several fragments can reach the
floor together. AA and reuse keep their dedicated preparation paths.

### Complete-producer birth storage

Eligible ordinary producers may keep each neighbor's born coordinates in a
contiguous `Vec<u32>`. Eligibility requires the entire candidate list, a job
whose conservative `2 * sum(raw candidate lengths)` node budget fits, and
`corpus.len() <= u32::MAX`. Every match emits at most two births, so a complete
job has at least two remaining logical nodes before each match. Its maximum
coordinate is less than the corpus length and therefore fits in `u32`.
Partial producers, AA, reuse and wide-coordinate corpora keep linked storage.
These are geometry and ownership conditions, not an input-type or alphabet test.

A group keeps two linked seed positions and promotes on its third birth; tiny
and ineligible groups allocate no vector. Compact coordinates can save space for
larger groups, but vector capacity and pool overhead affect the crossover.
Drain releases each promoted payload after encoding or pruning. Pool indices
stay unique until both direction drains finish; errors leave remaining payloads
owned by scratch. [Preparation storage](merge/prepare/mod.rs) documents the
field layout and allocation details next to their implementation.

The next batch's layout follows the preceding eligible tasks' birth shape.
`B` counts all admitted births, including zero-weight and later-pruned births;
`C_all` counts touched neighbor groups, including empty or removal-only groups.
Their ratio is a conservative mean group size. Successful commit updates the
policy using `B/C_all >= 16`; failures leave it unchanged, and restart resets
history. Linked eligible tasks also report shape so contiguous storage can be
re-enabled. With no eligible groups, the current policy stays unchanged.

A fresh attempt initially permits contiguous storage. The cutoff is an empirical
performance policy, independent of model semantics; it does not guarantee the
next batch's shape or a universal performance crossover. A kernel chosen before
collection keeps layout dispatch outside the position loop. See
[ContiguousBirthPolicy](merge/prepare/mod.rs) for policy and
[birth collection](merge/prepare/mod.rs) for measurement.

### Owner commit order

The index retains each owner's route, birth-grouping buffers, and completed-birth
vector between rounds. These buffers keep their largest capacity for the attempt.
Their live entries are cleared after all owner tasks join, including on an error;
route indices must not outlive the events they address. This avoids repeated
allocation while allowing unused capacity from an earlier round to remain resident.

Each changed owner performs these operations within one joined phase:

1. Stably group birth metadata while preserving the order of routed count actions.
2. Apply count actions in their original order. Fresh mode checks `u64` removals
   and retires counts below the floor; new keys receive their complete birth
   counts during publication. Reuse applies signed removals and additions, removing
   before adding for `Both`, and checks every intermediate value, including keys
   without a published cohort.
3. Publish complete births by moving their already encoded lists into fresh state.
4. Sum the remaining birth fragments, then encode and publish each complete key.
   Fresh mode applies the floor after reduction. Reuse uses the updated ledger
   and retains positive cohorts below the selection floor.
5. Refill the fresh owner's candidate prefix before returning, even if the owner
   had only count changes or complete births and no routed births.

In reuse mode, removal and birth cannot be combined into a net delta: intermediate
overflow can fail even when the net change is zero. [Index tests](pair_index.rs)
cover that ordering and direct plus partial publication in one commit.

One fragment vector is reused across keys and rule/direction buckets. Fresh
fragments occupy disjoint spatial runs and can be traversed directly in reverse.
Reuse left and right chains can interleave, so the encoder merges actual overlaps.
Zero-weight births with nonempty chains stay routed because the positions still
have an owner. Retained encoded lists allocate backing storage when needed.
Untouched owners skip commit work and refill lazily during selection.

### Failure behavior

Preparation errors return no plan to apply. Commit errors can occur after corpus
writes and partial shard updates. An error abandons the attempt; its index cannot
be resumed, and a round has no transactional rollback.

An active-ID restart also releases all attempt state. Only original input and the
chosen alphabet are reused. On completion or disposal, position owners are
dropped before their arena. Training corpus and worker scratch are released
before public model strings are built.

## Storage and resource costs

### Initial grouping

[Initial pair construction](initial_pairs.rs) emits bounded record waves.
Sixteen-bit initial IDs use eight-byte records; wider keys use twelve. Both
preserve complete pair identity and wave-local offsets. Stable grouping keeps
each key's physical order; the wave base restores full-width coordinates.
Weight runs and a unit-weight shortcut supply weighted counts. Generic multi-owner
routing uses dense directories and source-ordered disjoint counted slices at every
pool size. Fill is parallel; owners encode using the executing worker's resources.

When the plan has at most 256 distinct resolved initial IDs within u16, a bounded
pair collector can replace per-occurrence keyed records with four-byte wave-local
position offsets. Eligibility follows the IDs actually activated by retained
symbols, including affix aliases and zero-weight words; it does not follow a
pretokenizer flag or truncate vocabulary IDs. An ordinal lookup and sorted
original-ID table recover complete pair keys. Those keys retain the existing
owner routing and priority tie order. Stable counting partitions each pair's
positions by ascending spatial producer range, preserving occurrence order.

The collector's `n²` directory cost can outweigh the savings from four-byte
offsets on small sources. Admission bounds parallel metadata and compares its
estimated cost with the generic record working set. This is a selection heuristic,
not a bound on full training peak memory. Unknown alphabets, wide IDs, and sources
without a cheap exact edge count retain generic records. Exact budgets and cost
calculations live in [the bounded collector](initial_pairs/bounded.rs).

Admission compares the global edge count with the maximum wave's directory
estimate. A large source keeps the bounded collector even if its final wave is
small; that tail can pay a disproportionate directory cost rather than losing
the benefit of preceding large waves or requiring mixed record layouts.
`InitialCollector` owns the common wave lifecycle: each collector produces
grouped records, then shared encoding and publication build or append occurrence
lists. Both paths preserve checked weighted counts, floor handling, full pair
identity, and spatial order. Raw records retire before corpus materialization.

### Slot and coordinate storage

The corpus selects a 16-bit, packed 24-bit, or 32-bit slot type for its ID domain,
reserving a separator code. Packed 24-bit reads require an initialized guard slot
and the joined read/write protocol. The local contracts are in
[corpus/slots.rs](corpus/slots.rs).

Private coordinate storage keeps the low 32 bits beside each payload. Coordinates
that share one high half use that shared value; a separate high plane appears
when needed. This preserves full-width coordinates. Position buffers specialize
the same storage with an empty payload. Linked chains add bounded local node
references and reverse traversal.

### Encoded occurrence lists

[SortedPositions](storage/sorted_positions.rs) uses inline small cases and
unsigned LEB128 gaps. Each compressed group contains at most 128 positions: an
eight-byte absolute seed followed by gaps from preceding positions. A directory
of restart offsets lets a range cursor replay at most 127 gaps. Base 128 is the
integer codec's radix; the 128-position group size is a separate design choice.
This layout is private in-memory storage and does not define model serialization.

Append measures and checks its suffix before publishing the new count. A failed
append leaves the previous prefix readable. Small allocations use worker-exclusive
bump cursors below a fixed cutoff; large allocations remain individually owned.
Arena and heap ownership tags determine cleanup.

Compression saves space when nearby occurrences have small gaps. Iteration still
decodes those gaps, and seeking into a group must replay from its seed. Large gaps
need wider varints, while restart seeds and the directory add fixed per-group
overhead. Full-width positions are retained in every case.

### Arena and worker lifetimes

[Execution](execution.rs) owns the training pool, reusable ID directories, and
codec scratch. Task closures acquire and return directories, resetting touched
IDs on success and errors. Unwinding drops accumulator values and leaves empty
reusable state. Encoding resources belong to executing worker IDs, independently
of logical pair owners.

Published position lists borrow the arena, not the temporary `AllocationLease`
that grants exclusive access to one worker's allocation cursor. They can outlive
that lease. The arena must outlive its published lists, and lists and event chunks
must outlive the cursors and fragments that borrow them.

A task must not start nested pool work while holding worker resources: reentry
could attempt to acquire its own lock. Joined phases and owner-before-arena drop
order bound all uses of the storage.

### Local tradeoffs

Packed slots and compressed lists reduce memory use and add representation logic.
Delayed corpus construction reduces peak overlap with initial records. Arena and
directory reuse reduce repeated allocation. Complete producers avoid routing
birth chains and re-encoding them at the owner; split rules still need aggregation.
Rayon schedules work.

The radix port sorts complete pair keys and preserves incoming payload order
for equal keys. Its block path uses eight-byte compact or twelve-byte full-width
records. The fixed 512-record block uses 2 MiB or 3 MiB of scatter scratch,
respectively, plus nine bytes of metadata per input block and fixed directory
overhead. Small inputs use a separate sort with scratch proportional to input
size. Block-path metadata grows with `n / 512`. The reference paper's square-root space bound does
not apply to this fixed-block parameterization.

Performance results depend on the workload and measurement boundary. Follow the
[README measurement requirements](README.md#measurement-boundaries) to separate
feed, public `do_train`, and end-to-end costs. Store recorded results with their benchmark
inputs and build settings.

## Credits and references

Unsigned LEB128 gaps use base-128 varints, described in the
[Protocol Buffers integer encoding documentation](https://protobuf.dev/programming-guides/encoding/#base-128-varints).
That reference describes the integer encoding. The restart-list layout is a local
design, and the codec is not a source-code port of Protocol Buffers.

[storage/radix.rs](storage/radix.rs) is a Rust port of Robert Clausecker's
BSD-2-Clause
[`radixsort_permuted.c`](https://github.com/clausecker/radsort/blob/f69e816c3cd79d312cd67aea5b9cf1c338c1b371/radixsort_permuted.c)
at revision `f69e816c3cd79d312cd67aea5b9cf1c338c1b371`. The complete copyright and
license remain in the source. See the reference
[COPYING](https://github.com/clausecker/radsort/blob/f69e816c3cd79d312cd67aea5b9cf1c338c1b371/COPYING)
and the [Clausecker and Schintke paper](https://arxiv.org/abs/2607.05302).
