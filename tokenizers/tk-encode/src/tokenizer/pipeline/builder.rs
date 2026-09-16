use std::{collections::BTreeMap, sync::Arc};

use crate::{
    DecoderRuntime, PaddingParams, Result,
    pipeline::{
        NormalizerChain, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
        PipelinePreTokenizer, PipelineTokenizer,
    },
    vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary},
};

pub struct TokenizerBuilder {
    pub added_tokens: Vec<AddedToken>,
    pub normalizers: Vec<PipelineNormalizer>,
    pub pre_tokenizer: PipelinePreTokenizer,
    pub model: Arc<PipelineModel>,
    pub post_processor: PipelinePostProcessor,
    pub decoder: Option<DecoderRuntime>,
    /// Which token plays which role (`"eos_token"` -> `"</s>"`), so a `tokenizer.json` can carry
    /// the special-token metadata that used to need a separate `tokenizer_config.json`. Empty
    /// when the config declares none. `BTreeMap` so the writer emits a stable key order.
    pub role_to_token: BTreeMap<String, String>,
    /// Padding configuration. [`EncodeOptions::padding`] overrides it per call.
    pub padding: Option<PaddingParams>,
}

impl TokenizerBuilder {
    pub fn new(model: PipelineModel) -> Self {
        Self {
            added_tokens: vec![],
            decoder: None,
            normalizers: vec![],
            model: Arc::new(model),
            padding: None,
            pre_tokenizer: PipelinePreTokenizer::None,
            post_processor: PipelinePostProcessor::default(),
            role_to_token: BTreeMap::new(),
        }
    }

    pub fn build(self) -> Result<PipelineTokenizer> {
        let normalizers = NormalizerChain(match self.normalizers.last() {
            Some(PipelineNormalizer::Metaspace(_)) => {
                &self.normalizers[..self.normalizers.len() - 1]
            }
            _ => &self.normalizers[..],
        });

        let mut added_vocab = AddedVocabulary::new();
        added_vocab.add_tokens(
            self.added_tokens,
            self.model.vocab_size(),
            |token| self.model.token_to_id(token),
            Some(&normalizers),
        )?;
        Ok(PipelineTokenizer::from_parts(
            added_vocab,
            self.normalizers,
            self.pre_tokenizer,
            self.model,
            self.post_processor,
            self.decoder,
            self.role_to_token,
            self.padding,
        ))
    }
}

impl PipelineTokenizer {
    pub fn to_builder(&self) -> TokenizerBuilder {
        let mut added_tokens: Vec<_> = self
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .into_iter()
            .collect();
        added_tokens.sort_by_key(|&(id, _)| id);
        TokenizerBuilder {
            added_tokens: added_tokens.into_iter().map(|(_, token)| token).collect(),
            decoder: self.get_decoder().cloned(),
            model: self.inner.model.clone(),
            normalizers: self.get_normalizers().to_vec(),
            padding: self.get_padding().cloned(),
            post_processor: self.get_post_processor().clone(),
            pre_tokenizer: self.get_pre_tokenizer().clone(),
            role_to_token: self.get_role_to_token().clone(),
        }
    }
}
