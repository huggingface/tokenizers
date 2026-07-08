pub mod atom_tables;
pub mod classify;
pub mod fsm;
pub mod matchers;
pub mod patterns;
pub mod rules;
pub mod simd_classify;
#[cfg(target_arch = "x86_64")]
pub mod simd_avx_classify;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod simd_wasm_classify;
pub mod unicode;
pub mod unicode_tables;
