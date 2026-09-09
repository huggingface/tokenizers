# tokenizers-c

Experimental C bindings for the tokenizers library

Expect breaking changes in the ABI

## Build

```sh
cargo build --release
```

This produces `target/release/libtokenizers_c.{dylib,so,a}` and regenerates
`include/tokenizers/tokenizers.h` from `src/`.

## Use

```c
#include "tokenizers/tokenizers.h"
```

Compile with `-I<this dir>/include`, link with `-L<this dir>/target/release -ltokenizers_c`.
`examples/Makefile` has the exact flags, and `examples/quick_start.c` is the whole API in one
page: load, encode, read the ids, decode, free.

The header opens with the conventions every function follows (errors, ownership, NULL, threads,
text). Read that block first.

## Development

- `cargo test` runs the Rust-side unit tests.
- `make -C examples test` builds `quick_start.c` against the release dylib and runs it on the
  gpt2 fixture at `../../tokenizers/data/gpt2.json`, once normally and once against a missing
  file. `.github/workflows/c.yml` shows how CI fetches the fixture.
- `build.rs` fails the build when an exported function is not guarded against panics; the error
  message says what shape a guarded function has.
- `include/tokenizers/tokenizers.h` is checked in so consumers can vendor it. CI fails if it
  differs from what `cargo build` generates.
