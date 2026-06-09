use pcre2::bytes::RegexBuilder;

#[derive(Debug)]
pub struct SysRegex {
    regex: pcre2::bytes::Regex,
}

impl SysRegex {
    pub fn new(
        regex_str: &str,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let regex = RegexBuilder::new()
            .jit_if_available(true)
            .utf(true)
            .ucp(true)
            .build(regex_str)?;
        Ok(Self { regex })
    }

    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        Matches(self.regex.find_iter(inside.as_bytes()))
    }
}

pub struct Matches<'r, 't>(pcre2::bytes::Matches<'r, 't>);

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(Ok(mat)) => Some((mat.start(), mat.end())),
            None | Some(Err(_)) => None,
        }
    }
}
