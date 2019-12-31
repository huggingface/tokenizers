#
# Cargo helpers.
#

.PHONY : build
build :
	cargo build

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
	@./benches/run.sh

.PHONY : publish
publish :
	cargo publish
