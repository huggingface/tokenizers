use crate::Offsets;
use regex::Regex;

/// Pattern used to split a NormalizedString
pub trait Pattern {
    /// Slice the given string in a list of pattern match positions, with
    /// a boolean indicating whether this is a match or not.
    ///
    /// This method *must* cover the whole string in its outputs, with
    /// contiguous ordered slices.
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)>;
}

impl Pattern for char {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        todo!()
    }
}

impl Pattern for &str {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        todo!()
    }
}

impl Pattern for &Regex {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        todo!()
    }
}

impl<F> Pattern for F
where
    F: Fn(char) -> bool,
{
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        todo!()
    }
}
pub struct Invert<P: Pattern>(pub P);
/// When the given Regex matches words instead of the delimiter,
/// we use the `Reverse` helper to invert the delimiter flags
impl<P: Pattern> Pattern for Invert<P> {
    fn find_matches(&self, inside: &str) -> Vec<(Offsets, bool)> {
        todo!()
    }
}
