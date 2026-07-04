use std::str::from_utf8_unchecked;

use unicode_categories::UnicodeCategories;
pub trait Matcher {
    fn run_end(b: &[u8], from: usize) -> usize;
}
pub struct Letter;
impl Matcher for Letter {
    #[inline]
    fn run_end(b: &[u8], from: usize) -> usize {
        // SAFETY: from has to always be a char boundary (loop advances by whole tokens)
        let s = unsafe { from_utf8_unchecked(&b[from..]) };
        for (off, c) in s.char_indices() {
            if !c.is_letter() {
                return from + off;
            }
        }
        b.len()
    }
}
