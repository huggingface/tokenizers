DATA_DIR = data

dir_guard=@mkdir -p $(@D)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2-merges.txt $(DATA_DIR)/gpt2.json $(DATA_DIR)/bert-base-uncased-vocab.txt $(DATA_DIR)/big.txt $(DATA_DIR)/small.txt $(DATA_DIR)/albert-base-v1-tokenizer.json  $(DATA_DIR)/llama-3-tokenizer.json
# Exactly what tk-serialize/benches/{encode,decode}.rs read: big.txt and llama-3 come from
# SHARED_RESOURCES, plus the Japanese corpus decode uses and the config encode profiles.
BENCHMARK_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/deepseek-v4.json
TESTS_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram.json $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json $(DATA_DIR)/fixtures/modalities/agentic-traces.txt

# One language per encode regime (script structure × vocab coverage), not one per
# script: eng is the vocab-native anchor, rus the 2-byte European alphabet, arb the
# 2-byte RTL abjad, hin mainstream Indic, tam the extreme-fertility stress probe,
# amh the poorly-covered 3-byte script, cmn the Han inventory, jpn script mixing,
# tha spaceless text. Structural near-twins (ell/heb/ben/kat/kor) were cut: they
# tracked their representative on every model and only diluted the aggregates.
FIXTURE_LANGS = amh_Ethi arb_Arab cmn_Hani eng_Latn hin_Deva jpn_Jpan rus_Cyrl tam_Taml tha_Thai
# Only the sparse added-token corpora: realistic marker density (text sprinkled with
# specials, like chat-template traffic), covering both the raw-special and the
# normalized-added scan paths. The dense variants were marker soup that skewed every
# aggregate while measuring nothing the sparse ones don't.
FIXTURE_MODALITIES = agentic_swe agentic-traces code_mixed math_latex added_special_sparse added_normalized_sparse
FIXTURES_RESOURCES = $(FIXTURE_LANGS:%=$(DATA_DIR)/fixtures/lang/%.txt) $(FIXTURE_MODALITIES:%=$(DATA_DIR)/fixtures/modalities/%.txt)

.PHONY : build
build :
	cargo build --workspace --all-targets

.PHONY : release
release :
	cargo build --workspace --release

.PHONY : format
format :
	cargo fmt --all --

.PHONY : lint
lint :
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings

.PHONY : test
test : $(TESTS_RESOURCES)
	cargo test --workspace

.PHONY : doc
doc :
	cargo doc --workspace

.PHONY : publish
publish :
	cargo publish

# The features that are `cfg` gates on enum variants, so a combination of them can break on its own.
MODEL_FEATURES = bpe,unigram,wordpiece,wordlevel,normalizers

.PHONY : all-checks
all-checks : lint test doc feature-matrix

# Compile, lint and test tk-encode under each meaningful feature combination.
#
# In `all-checks` because nothing else here can catch a combination breaking:
# `lint` is --all-features and `test` is the default set, so both are blind to
# the builds in between, where the per-model and `normalizers` cfg gates
# actually interact.
# Needs cargo-hack: `cargo install cargo-hack`.
.PHONY : feature-matrix
feature-matrix : $(SHARED_RESOURCES)
	cargo hack --package tk-encode --feature-powerset --depth 2 \
	  --include-features $(MODEL_FEATURES) clippy --all-targets -- -D warnings
	cargo hack --package tk-encode --each-feature \
	  --include-features $(MODEL_FEATURES) test --lib

# tk-encode's stripped and gzipped size per feature configuration, under
# `--profile minsize`.
#
# Deliberately NOT in `all-checks`: there is no agreed size budget, so this can
# only ever print numbers -- it has no pass/fail to contribute -- and it pays for
# a whole separate `target/minsize` tree (one fat-LTO link per rung) to print
# them. Run it when changing what gets compiled in, not on every check.
.PHONY : slim-size
# Gzipped is the only honest number: Mach-O segments are 16 KiB-quantised, so on-disk size cannot
# resolve anything under ~16 KB (it reported 0 for a change that really removed 1,504 bytes). And
# the largest data object in the binary is a zero-filled array: 9.4% of the file, 0.1% of the
# download. Decide against the compressed size.
slim-size :
	cargo build --profile minsize -p tk-serialize --example binsize_pipeline --no-default-features --features bpe,deserialize
	@gzip -9 -c target/minsize/examples/binsize_pipeline | wc -c | tr -d ' ' \
	  | xargs printf 'slim tk-encode + tk-serialize: %s bytes gzipped\n'

# The host triple has to be named explicitly: `-Z build-std` only applies to an explicit --target,
# and without one cargo reuses the precompiled libstd and the flags silently do nothing.
HOST_TRIPLE := $(shell rustc -vV | awk '/^host:/ {print $$2}')
# Overridable, because the failure this guards against is a *stale* `nightly`, and re-pointing the
# default alias is not always the caller's call: `make slimest NIGHTLY=nightly-2026-04-28`.
NIGHTLY ?= nightly
SLIMEST_FLAGS = -Z build-std=std,panic_abort --target $(HOST_TRIPLE)

# `panic_immediate_abort` used to be a `-Z build-std-features` value. It is now a real panic
# strategy and the old spelling is a hard error:
#   error: panic_immediate_abort is now a real panic strategy! Enable it with `panic =
#   "immediate-abort"` in Cargo.toml, or with the compiler flags `-Zunstable-options
#   -Cpanic=immediate-abort`. In both cases, you still need to build core, e.g. with `-Zbuild-std`
# So it has to reach rustc as a codegen flag, and `-C panic` only accepts the new value under
# `-Zunstable-options`.
#
# Deliberately RUSTFLAGS and not `panic = "immediate-abort"` in `[profile.slimest]`: a profile value
# that stable cargo cannot parse would break *every* cargo command in the workspace (`make build`,
# `make test`, `cargo metadata`, CI on stable), not just this target. The cost of keeping it here is
# that it replaces any RUSTFLAGS already in the environment for this one recipe.
SLIMEST_RUSTFLAGS = -Zunstable-options -Cpanic=immediate-abort

.PHONY : slimest
# The smallest build there is: `minsize` plus a `std` rebuilt without panic, unwinding or backtrace
# symbolisation. Needs nightly and the rust-src component:
#   rustup toolchain install nightly && rustup component add rust-src --toolchain nightly
#
# `rustup update nightly` if this fails on the panic strategy (see SLIMEST_RUSTFLAGS): a nightly
# older than the rename only accepts the removed `-Z build-std-features` spelling. Verified on
# 1.97.0-nightly (2026-04-27); 1.89.0-nightly (2025-06-22) is too old.
#
# Same feature set as `slim-size` on purpose, so the two numbers differ by the profile alone.
# Leaving `bpe` off here made the rungs look like they measured different programs, which is how a
# reviewer concluded the 328k -> 195k drop partly came from dropping BPE. It does not: `bpe` is an
# empty feature (`bpe = []` in tk-encode) and no `cfg` in the workspace reads it, so BPE is compiled
# either way. Adding it does move the gzipped size by 24 bytes -- reproducibly -- but that is the
# crate metadata hash changing symbol names, not code. Worth 24 bytes to stop the ladder inviting
# that misreading.
slimest :
	RUSTFLAGS="$(SLIMEST_RUSTFLAGS)" cargo +$(NIGHTLY) build --profile slimest $(SLIMEST_FLAGS) \
	  -p tk-serialize --example binsize_pipeline --no-default-features --features bpe,deserialize

.PHONY : slimest-size
# Same, and print the gzipped size next to `slim-size`'s so the difference is visible.
slimest-size : slimest
	@b=target/$(HOST_TRIPLE)/slimest/examples/binsize_pipeline; \
	  gzip -9 -c $$b | wc -c | tr -d ' ' | xargs printf 'slimest tk-encode: %s bytes gzipped\n'

# The two benches live in tk-serialize, which is the only crate that can build a
# `PipelineTokenizer`. The big cross-engine benchmarks live in huggingface/tokbench.
.PHONY : bench
bench : $(BENCHMARK_RESOURCES)
	cargo bench -p tk-serialize

# Every file the tests and benches read (corpora, vocabs, tokenizer configs),
# plus the fixture corpora.
.PHONY : data
data : $(TESTS_RESOURCES) $(BENCHMARK_RESOURCES) $(FIXTURES_RESOURCES)

# Multilingual + modality corpora for cross-language benchmarks; see
# fixtures/FIXTURES.md in $(HF_TEST_REPO) for provenance.
.PHONY : fixtures
fixtures : $(FIXTURES_RESOURCES)

HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
# Fetch fixtures through the huggingface_hub CLI: it handles auth (HF_TOKEN),
# retry/backoff on rate limits, and caching for us. CI overrides HF with
# `uvx --from huggingface_hub hf` to avoid a separate install step.
HF ?= hf
# Pinned, not `main`. `json_oracle`'s recorded digests are only meaningful against a fixed fixture
# set, and the `$(DATA_DIR)/%` rule below is existence-based -- it never re-downloads a file you
# already have. Together those two facts silently pinned one developer's digests to a stale
# `llama-2.json` and produced four spurious drifts on any fresh checkout. Bump this deliberately,
# and re-record the digests in the same commit.
HF_REVISION ?= e20ac5914f5ada06874e4f9d45e49b31a049904e

$(DATA_DIR)/% :
	$(dir_guard)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --revision $(HF_REVISION) --local-dir $(DATA_DIR)
