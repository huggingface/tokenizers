BENCHMARK_DIR = benches
BENCHMARK_RESOURCES = $(BENCHMARK_DIR)/gpt2-vocab.json $(BENCHMARK_DIR)/gpt2-merges.txt $(BENCHMARK_DIR)/big.txt

.PHONY : build
build :
	cargo build --all-targets

.PHONY : release
release :
	cargo build --release

.PHONY : format
format :
	cargo fmt --

.PHONY : lint
lint :
	cargo fmt -- --check
	cargo fmt -- $(BENCHMARK_DIR)/*.rs --check
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY : test
test :
	cargo test

.PHONY : doc
doc :
	cargo doc

.PHONY : publish
publish :
	cargo publish

.PHONY : all-checks
all-checks : lint test doc

.PHONY : bench
bench : $(BENCHMARK_RESOURCES)
	cargo bench -- --verbose

$(BENCHMARK_DIR)/gpt2-% :
	wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-$* -O $@

$(BENCHMARK_DIR)/big.txt :
	wget https://norvig.com/big.txt -O $@
