//! Regenerate fast_split's committed classify tables:
//!   cargo run -p bitmap_gen [-- <out_path>]
//! Default `out_path` = ../fast_split/src/atom_tables.rs. `generate_atom_tables` self-validates every
//! codepoint against the reference `atom()`, so an inconsistent scheme change fails HERE, not at ship.
fn main() {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../fast_split/src/atom_tables.rs");
    let out = std::env::args().nth(1).unwrap_or_else(|| default.to_string());
    std::fs::write(&out, bitmap_gen::generate_atom_tables()).expect("write atom_tables.rs");
    eprintln!("bitmap_gen: wrote {out}");
}
