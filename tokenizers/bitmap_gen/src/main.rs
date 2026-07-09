//! Regenerate fast_split's committed classify tables:
//!   cargo run -p bitmap_gen [-- <atom_out> [<script_out>]]
//!   cargo run -p bitmap_gen --features scripts        # also writes script_tables.rs
//! Defaults: `../fast_split/src/atom_tables.rs` and (under `scripts`) `.../src/script_tables.rs`. Each
//! generator self-validates every codepoint against its reference, so an inconsistent scheme change
//! fails HERE, not at ship.
fn main() {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../fast_split/src/atom_tables.rs");
    let out = std::env::args().nth(1).unwrap_or_else(|| default.to_string());
    std::fs::write(&out, bitmap_gen::generate_atom_tables()).expect("write atom_tables.rs");
    eprintln!("bitmap_gen: wrote {out}");

    #[cfg(feature = "scripts")]
    {
        let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../fast_split/src/script_tables.rs");
        let out = std::env::args().nth(2).unwrap_or_else(|| default.to_string());
        std::fs::write(&out, bitmap_gen::generate_script_tables()).expect("write script_tables.rs");
        eprintln!("bitmap_gen: wrote {out}");
    }
}
