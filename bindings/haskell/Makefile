all: FORCE
	cargo build --release
	rm -f tokenizers-haskell/lib/libtokenizers_haskell.*
	rsync -arv ../../target/release/libtokenizers_haskell.* tokenizers-haskell/lib/.
	cd tokenizers-haskell; make all

watch: FORCE
	cargo watch -x "build --release"

FORCE: ;

