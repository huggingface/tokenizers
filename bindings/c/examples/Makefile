CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
INCLUDE_DIR := ../include
LIB_DIR := ../target/release
FIXTURE := ../../../tokenizers/data/gpt2.json
BAD_FIXTURE := does-not-exist.json

# The dylib mirrors how a real consumer would link tokenizers-c: build it once with
# `cargo build --release` in .., then link against the shared library like any other
# precompiled C library.
quick_start: quick_start.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(LIB_DIR) -ltokenizers_c -Wl,-rpath,$(LIB_DIR)

run: quick_start
	./quick_start $(FIXTURE) "Hello, world!"

# A bad tokenizer path must surface as a clean TkError and exit 1, not crash. The leading `!`
# inverts make's exit-code check, since quick_start is expected to exit non-zero here.
run_error: quick_start
	! ./quick_start $(BAD_FIXTURE) "Hello, world!"

test: run run_error

format:
	$(CLANG_FORMAT) -i quick_start.c

format_check:
	$(CLANG_FORMAT) --dry-run --Werror quick_start.c

lint:
	$(CLANG_TIDY) --warnings-as-errors='*' quick_start.c -- $(CFLAGS) -I$(INCLUDE_DIR)

clean:
	rm -f quick_start

.PHONY: run run_error test clean format format_check lint
