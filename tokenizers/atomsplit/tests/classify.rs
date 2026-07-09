//! Integration test for the atom classifier. Kept out of `src/` so the core stays production-only.
use atomsplit::classify::{classify, classify_scalar, Atoms};

/// The byte-exactness gate (spec §8): the SIMD path (NEON on aarch64) MUST equal the scalar walk.
#[test]
fn simd_matches_scalar_atoms() {
    let unit = "Hello, 世界! ½ + ٠١ Ⅷ café\tнаука ไทย 😀\u{0301}mark _u 'q' ©s ½²¼ 안녕 ";
    let corpus = unit.repeat(40); // >32 B so the SIMD chunk loop actually runs
    let text = corpus.as_bytes();
    let mut simd = vec![0u8; text.len()];
    let mut scalar = vec![0u8; text.len()];
    classify::<Atoms>(text, &mut simd);
    classify_scalar::<Atoms>(text, &mut scalar);
    assert_eq!(simd, scalar, "SIMD classify must be byte-exact vs scalar");
}
