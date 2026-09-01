CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
INCLUDE_DIR := ../include
LIB_DIR := ../target/release
FIXTURE := ../../../tokenizers/data/gpt2.json
BAD_FIXTURE := does-not-exist.json
SANITIZE_FLAGS := -fsanitize=address,undefined -g

# The dylib mirrors how a real consumer would link tokenizers-c: build it once with
# `cargo build --release` in .., then link against the shared library like any other
# precompiled C library.
encode: encode.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(LIB_DIR) -ltokenizers_c -Wl,-rpath,$(LIB_DIR)

# Only encode.c is compiled with the sanitizers; ASan still intercepts every malloc/free the
# dylib makes, since they resolve to the same libc symbols in the one process.
encode-sanitized: encode.c
	$(CC) $(CFLAGS) $(SANITIZE_FLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(LIB_DIR) -ltokenizers_c -Wl,-rpath,$(LIB_DIR)

null-safety: null_safety.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(LIB_DIR) -ltokenizers_c -Wl,-rpath,$(LIB_DIR)

# Same rationale as encode-sanitized: ASan/UBSan still see every malloc/free the dylib makes.
# Most valuable here for the double-free-on-the-same-handle case, which is UB that a plain
# build isn't guaranteed to crash on even when it's wrong.
null-safety-sanitized: null_safety.c
	$(CC) $(CFLAGS) $(SANITIZE_FLAGS) -I$(INCLUDE_DIR) -o $@ $< -L$(LIB_DIR) -ltokenizers_c -Wl,-rpath,$(LIB_DIR)

run: encode
	./encode $(FIXTURE) "Hello, world!"

# A bad tokenizer path must surface as a clean TkError and exit 1, not crash. The leading `!`
# inverts make's exit-code check, since encode is expected to exit non-zero here.
run-error: encode
	! ./encode $(BAD_FIXTURE) "Hello, world!"

run-sanitized: encode-sanitized
	./encode-sanitized $(FIXTURE) "Hello, world!"
	! ./encode-sanitized $(BAD_FIXTURE) "Hello, world!"

# Every NULL-argument case across the C ABI must come back as a clean TkError (or a safe
# no-op for tk_*_free), never a crash. See null_safety.c.
run-null-safety: null-safety
	./null-safety $(FIXTURE)

run-null-safety-sanitized: null-safety-sanitized
	./null-safety-sanitized $(FIXTURE)

test: run run-error run-sanitized run-null-safety run-null-safety-sanitized

format:
	$(CLANG_FORMAT) -i encode.c null_safety.c

format-check:
	$(CLANG_FORMAT) --dry-run --Werror encode.c null_safety.c

lint:
	$(CLANG_TIDY) --warnings-as-errors='*' encode.c null_safety.c -- $(CFLAGS) -I$(INCLUDE_DIR)

clean:
	rm -f encode encode-sanitized null-safety null-safety-sanitized

.PHONY: run run-error run-sanitized run-null-safety run-null-safety-sanitized test clean \
	format format-check lint
