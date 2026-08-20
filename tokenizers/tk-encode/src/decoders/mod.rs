pub mod bpe;
pub mod byte_fallback;
pub mod byte_level;
pub mod ctc;
pub mod fuse;
pub mod metaspace;
pub mod replace;
pub mod runtime;
#[cfg(feature = "serde")]
mod serialization;
pub mod strip;
pub mod wordpiece;

pub use runtime::DecoderRuntime;

// Every decoder here is a type of its own, including the three that used to be borrowed from
// elsewhere: `byte_level` and `metaspace` were `pub use`d straight out of `pre_tokenizers`, and
// `Replace` was taken from `normalizers`. One type playing two roles meant one variant sitting in
// two wrappers, and a variant in two wrappers cannot be given one on-disk shape without also giving
// it the other's. A decoder is now a decoder and nothing else.
//
// `DecoderWrapper` and its hand-written `Deserialize` (tagged, with an untagged legacy fallback)
// are gone, deleted with the config layer. A `Sequence` decoder needs no wrapper to be
// parameterised by: it is `DecoderRuntime::Sequence(Vec<DecoderRuntime>)`, which `tk-serialize`
// reads and writes directly. What the encode path needs is [`DecoderRuntime`] and nothing else.
