#
# Cargo helpers.
#

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
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY : test
test :
	cargo test

.PHONY : doc
doc :
	cargo doc

.PHONY : all-checks
all-checks : lint test doc

.PHONY : bench
bench :
	@echo 'Updating benchmark fixtures...'
	@test -f ./benches/gpt2-vocab.json \
		&& echo 'gpt2-vocab.json up-to-date' \
		|| wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O ./benches/gpt2-vocab.json
	@test -f ./benches/gpt2-merges.txt \
		&& echo 'gpt2-merges.txt up-to-date' \
		|| wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O ./benches/gpt2-merges.txt
	@test -f ./benches/big.txt \
		&& echo 'big.txt up-to-date' \
		|| wget https://norvig.com/big.txt -O ./benches/big.txt
	cargo bench -- --verbose

.PHONY : publish
publish :
	cargo publish
