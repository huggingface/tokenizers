JAVAC ?= javac
JAVA ?= java
LIB_DIR := ../c/target/release
FIXTURE := ../../tokenizers/data/gpt2.json

# Needs libtokenizers_c: `cargo build --release` in ../c first.
build:
	$(JAVAC) --release 22 -d build src/co/huggingface/tokenizers/Tokenizer.java test/TokenizerTest.java

test: build
	$(JAVA) -ea --enable-native-access=ALL-UNNAMED -Djava.library.path=$(LIB_DIR) -cp build TokenizerTest $(FIXTURE)

clean:
	rm -rf build

.PHONY: build test clean
