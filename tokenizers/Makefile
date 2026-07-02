DATA_DIR = data
BENCHMARK_DIR = benches
TESTS_DIR = tests

dir_guard=@mkdir -p $(@D)

SHARED_RESOURCES = $(DATA_DIR)/gpt2-vocab.json $(DATA_DIR)/gpt2-merges.txt $(DATA_DIR)/bert-base-uncased-vocab.txt $(DATA_DIR)/big.txt $(DATA_DIR)/small.txt $(DATA_DIR)/albert-base-v1-tokenizer.json  $(DATA_DIR)/llama-3-tokenizer.json
BENCHMARK_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/bert-wiki.json $(DATA_DIR)/fixtures/modalities/agentic-traces.txt
TESTS_RESOURCES = $(SHARED_RESOURCES) $(DATA_DIR)/unigram.json $(DATA_DIR)/unigram_wagahaiwa_nekodearu.txt $(DATA_DIR)/roberta.json $(DATA_DIR)/tokenizer-wiki.json $(DATA_DIR)/bert-wiki.json $(DATA_DIR)/fixtures/modalities/agentic-traces.txt

FIXTURE_LANGS = amh_Ethi arb_Arab ben_Beng cmn_Hani ell_Grek eng_Latn heb_Hebr hin_Deva jpn_Jpan kat_Geor kor_Hang rus_Cyrl tam_Taml tha_Thai
FIXTURE_MODALITIES = agentic_swe agentic-traces code_mixed math_latex
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

# Multilingual + modality corpora for cross-language benchmarks; see
# fixtures/FIXTURES.md in $(HF_TEST_REPO) for provenance.
.PHONY : fixtures
fixtures : $(FIXTURES_RESOURCES)

HF_TEST_REPO = hf-internal-testing/tokenizers-test-data
# Fetch fixtures through the huggingface_hub CLI: it handles auth (HF_TOKEN),
# retry/backoff on rate limits, and caching for us. CI overrides HF with
# `uvx --from huggingface_hub hf` to avoid a separate install step.
HF ?= hf
HF_REVISION ?= main

$(DATA_DIR)/% :
	$(dir_guard)
	$(HF) download $(HF_TEST_REPO) $* --repo-type dataset --revision $(HF_REVISION) --local-dir $(DATA_DIR)
