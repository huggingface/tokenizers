//! Fast-path token structures for the experimental `PipelineTokenizer`, parked here until the
//! pipeline PR wires them in: the MPHF [`bucket_vocab_store::BucketVocabStore`], the special-token
//! [`buckets::Buckets`] matcher, and the bucket-based [`bucket_added_vocabulary::AddedVocabulary`].
//! The legacy `Tokenizer` path uses `crate::tokenizer::AddedVocabulary` instead; these are not
//! re-exported at the crate root so they don't collide with it.
pub mod bucket_added_vocabulary;
pub mod bucket_vocab_store;
pub mod buckets;
