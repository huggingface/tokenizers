/// The default capacity of a model's [`WordCache`](crate::utils::word_cache::WordCache).
pub static DEFAULT_CACHE_CAPACITY: usize = 65_536;
/// The maximum length we should cache in a model
/// Strings that are too long have minimal chances to cache hit anyway
pub static MAX_LENGTH: usize = 256;
