use compact_str::{CompactString, ToCompactString};

pub fn to_compact_strings_option(
    input_options: Option<impl ToCompactString>,
) -> Option<CompactString> {
    input_options.map(|s| s.to_compact_string())
}

pub fn to_compact_strings_result<E>(
    input_results: Result<impl ToCompactString, E>,
) -> Result<CompactString, E> {
    input_results.map(|s| s.to_compact_string())
}

pub fn to_compact_strings<T: ToCompactString, C: IntoIterator<Item = T>>(
    input_strings: C,
) -> Vec<CompactString> {
    input_strings
        .into_iter()
        .map(|s| s.to_compact_string())
        .collect()
}
