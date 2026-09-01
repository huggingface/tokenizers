CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra
INCLUDE_DIR := ../include
LIB_DIR := ../target/release
FIXTURE := ../../../tokenizers/data/gpt2.json

# The dylib mirrors how a real consumer would link tokenizers-c: build it once with
# `cargo build --release` in .., then link against the shared library like any other
# precompiled C library.
encode: encode.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(LIB_DIR) -ltokenizers_c -Wl,-rpath,$(LIB_DIR)

run: encode
	./encode $(FIXTURE) "Hello, world!"

clean:
	rm -f encode

.PHONY: run clean
