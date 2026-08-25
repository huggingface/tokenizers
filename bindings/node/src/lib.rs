#![deny(clippy::all)]

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod pipeline;
