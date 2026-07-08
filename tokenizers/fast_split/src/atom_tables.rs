//! Shared atom classify tables (dense) — baked at compile time by `build.rs` (the `bitmap_gen`
//! generator; build-dep `unicode_categories`, not a runtime dep). Both the SIMD kernel and the scalar
//! reader `Tables::classify_char` index `ATOM_TABLES`. See TAG_CLASSIFY_SPEC.md §1/§7.
include!(concat!(env!("OUT_DIR"), "/atom_tables.rs"));
