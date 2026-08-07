DATA_DIR = data
BENCHMARK_DIR = benches
TESTS_DIR = tests

dir_guard=@mkdir -p $(@D)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2-merges.txt $(DATA_DIR)/gpt2.json $(DATA_DIR)/bert-base-uncased-vocab.txt $(DATA_DIR)/big.txt $(DATA_DIR)/small.txt $(DATA_DIR)/albert-base-v1-tokenizer.json  $(DATA_DIR)/llama-3-tokenizer.json
BENCHMARK_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/bert-wiki.json $(DATA_DIR)/fixtures/modalities/agentic-traces.txt
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

# Models benched by tk-encode/examples/fixture_bench.rs; the manifest is the
# single source of truth (Rust reads it too).
BENCH_MODELS = tk-encode/examples/bench_models.json

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
	cargo fmt -- $(BENCHMARK_DIR)/*.rs --check
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

.PHONY : all-checks
all-checks : lint test doc

.PHONY : bench
bench : $(BENCHMARK_RESOURCES)
	cargo bench -- --verbose

# Every file the tests and benches read (corpora, vocabs, tokenizer configs),
# plus the fixture corpora. Model tokenizers from the bench manifest have
# their own target: `models`.
.PHONY : data
data : $(TESTS_RESOURCES) $(BENCHMARK_RESOURCES) $(FIXTURES_RESOURCES)

# Multilingual + modality corpora for cross-language benchmarks; see
# fixtures/FIXTURES.md in $(HF_TEST_REPO) for provenance.
.PHONY : fixtures
fixtures : $(FIXTURES_RESOURCES)

# Fetch each bench model's tokenizer config (the `file` listed in the manifest)
# from the test-data dataset via the generic data/% rule. Same controlled
# mirror as the fixtures, so no direct pulls from model repos (supply chain).
.PHONY : models
models :
	@python3 -c "import json; [print(m.get('file') or m['name'] + '.json') for m in json.load(open('$(BENCH_MODELS)'))]" \
	| while read -r file; do \
	    $(MAKE) $(DATA_DIR)/$$file; \
	  done

HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
# Fetch fixtures through the huggingface_hub CLI: it handles auth (HF_TOKEN),
# retry/backoff on rate limits, and caching for us. CI overrides HF with
# `uvx --from huggingface_hub hf` to avoid a separate install step.
HF ?= hf
HF_REVISION ?= main

$(DATA_DIR)/% :
	$(dir_guard)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --revision $(HF_REVISION) --local-dir $(DATA_DIR)
