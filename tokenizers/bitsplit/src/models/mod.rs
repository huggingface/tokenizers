//! One module per pre-tokenization **regex**, each unrolled against the primitives in
//! [`crate`]. Models that ship the same regex share a module — o200k covers Llama-4, gpt-oss and
//! MiniMax-M2; cl100k covers Llama-3, GLM-4.6 and (at digit cap 1) Qwen.
//!
//! Deliberately not one parameterised grammar: folding these together made every fix a three-way
//! risk and none of them could be read on its own.
pub mod cl100k;
pub mod deepseek;
pub mod gpt2;
pub mod kimi;
pub mod o200k;
pub mod tekken;
