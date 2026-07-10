//! Regenerate atomsplit's committed classify tables:
//!   cargo run -p bitmap_gen
//! Writes ../atomsplit/src/atom_tables.rs (the pretokenizer atom alphabet) and
//! ../atomsplit/src/norm_tables.rs (the normalization property bitmask). Both self-validate every
//! codepoint against their reference classifier, so an inconsistent scheme change fails HERE, not at ship.
fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../atomsplit/src");
    for (name, src) in [
        ("atom_tables.rs", bitmap_gen::generate_atom_tables()),
        ("norm_tables.rs", bitmap_gen::generate_norm_tables()),
    ] {
        let out = format!("{dir}/{name}");
        std::fs::write(&out, src).unwrap_or_else(|e| panic!("write {name}: {e}"));
        eprintln!("bitmap_gen: wrote {out}");
    }
}
