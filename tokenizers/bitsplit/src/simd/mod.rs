//! Per-arch kernels. Every one of these is PURE PERF and byte-exact with a portable path that is
//! always compiled, so correctness never depends on a kernel being present:
//!   - [`neon`] / [`x86`] build the block's bitstreams; `build_block_scalar` is the reference.
//!
//! (`classify` has its own kernels, next to the tables they index.)
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;
