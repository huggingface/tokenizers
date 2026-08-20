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
// `DecoderWrapper`, its hand-written `Deserialize` (tagged, with an untagged legacy fallback) and
// the `Sequence` decoder that holds a `Vec<DecoderWrapper>` all live in `tk-convert`: the wrapper
// exists to be deserialized, and a `Vec` of it can only be parameterised where the wrapper is.
// What the encode path needs is [`DecoderRuntime`].
