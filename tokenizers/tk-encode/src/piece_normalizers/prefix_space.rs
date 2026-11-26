/// Writes a space at the start of a piece that does not start with one.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct PrefixSpace;

impl PrefixSpace {
    /// Appends `piece` to `out`, with a space in front unless `piece` is empty or starts with one.
    pub fn normalize_into(&self, piece: &str, out: &mut String) {
        if !piece.is_empty() && !piece.starts_with(' ') {
            out.push(' ');
        }
        out.push_str(piece);
    }
}
