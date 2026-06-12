#![deny(clippy::all)]

// Use mimalloc as the global allocator inside this cdylib (the Rust
// "binary" loaded by Node).  Replaces glibc malloc for every
// allocation made by Rust code inside tokenizers; cuts the per-call
// malloc/free LSE atomic traffic that shows up on aarch64.  The
// choice is local to this cdylib and does not affect the
// `tokenizers` library crate or any Rust binary that consumes it
// directly.
//
// The same cfg expression as the target-conditional dependency in
// `Cargo.toml`.  Keep these in sync when adding platforms.
#[cfg(all(
  feature = "mimalloc",
  any(
    all(
      target_os = "linux",
      any(target_arch = "x86_64", target_arch = "aarch64"),
      target_env = "gnu"
    ),
    all(target_os = "macos", target_arch = "aarch64"),
  )
))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

mod arc_rwlock_serde;
pub mod decoders;
pub mod encoding;
pub mod models;
pub mod normalizers;
pub mod pre_tokenizers;
pub mod processors;
pub mod tasks;
pub mod tokenizer;
pub mod trainers;
pub mod utils;
