//! One module per pre-tokenization **regex**, each unrolled against the primitives in
//! [`crate`]. Models that ship the same regex share a module — o200k covers Llama-4, gpt-oss and
//! MiniMax-M2; cl100k covers Llama-3, GLM-4.6 and (at digit cap 1) Qwen.
//!
//! o200k, tekken and kimi ARE one parameterised grammar ([`family_o200k`]) — they were three files whose
//! bitstream halves were 97%/91% identical and whose atom table, decode and entire scalar escape
//! were byte-identical, so a fix had to be applied three times to be a fix at all. The three knobs
//! that actually differ are named in `family_o200k`'s header. deepseek, cl100k and gpt2 stay separate:
//! their grammars genuinely diverge.
pub mod cl100k;
pub mod deepseek;
mod family_gpt;
mod family_o200k;
pub mod gpt2;
pub mod kimi;
pub mod o200k;
pub mod tekken;
