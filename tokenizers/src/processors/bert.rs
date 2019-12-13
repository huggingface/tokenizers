use crate::tokenizer::{Encoding, PostProcessor, Result};
use crate::utils::{truncate_encodings, TruncationStrategy};

struct BertProcessing {
    max_len: usize,
    trunc_strategy: TruncationStrategy,
    trunc_stride: usize,

    sep: (String, u32),
    cls: (String, u32),
}

impl PostProcessor for BertProcessing {
    fn process(&self, encoding: Encoding, pair_encoding: Option<Encoding>) -> Result<Encoding> {
        let special_token_len = if pair_encoding.is_some() { 3 } else { 2 };
        let total_len = encoding.get_ids().len()
            + pair_encoding
                .as_ref()
                .map(|e| e.get_ids().len())
                .unwrap_or(0)
            + special_token_len;

        let need_trunc = total_len - self.max_len;
        println!("Need to trunc {}", need_trunc);
        let (mut encoding, pair_encoding) = truncate_encodings(
            encoding,
            pair_encoding,
            need_trunc,
            self.trunc_strategy,
            self.trunc_stride,
        )?;

        // Prepare ids
        let ids = [&[self.cls.1], &encoding.get_ids()[..], &[self.sep.1]].concat();
        let pair_ids = pair_encoding
            .as_ref()
            .map(|encoding| [&encoding.get_ids()[..], &[self.sep.1]].concat());

        // Prepare tokens
        let tokens = [
            &[self.cls.0.clone()],
            &encoding.get_tokens()[..],
            &[self.sep.0.clone()],
        ]
        .concat();
        let pair_tokens = pair_encoding
            .as_ref()
            .map(|encoding| [&encoding.get_tokens()[..], &[self.sep.0.clone()]].concat());

        // Prepare offsets
        let offsets = [&[(0, 0)], &encoding.get_offsets()[..], &[(0, 0)]].concat();
        let pair_offsets = pair_encoding
            .as_ref()
            .map(|encoding| [&encoding.get_offsets()[..], &[(0, 0)]].concat());

        // Prepare type ids
        let type_ids = [&[0], &encoding.get_type_ids()[..], &[0]].concat();
        let pair_type_ids = pair_encoding
            .as_ref()
            .map(|encoding| [&encoding.get_type_ids()[..], &[1]].concat());

        let special_tokens = [&[1u32], &vec![0; encoding.get_ids().len()][..], &[1]].concat();
        let pair_special_tokens = pair_encoding
            .as_ref()
            .map(|encoding| [&vec![0u32; encoding.get_type_ids().len()][..], &[1]].concat());

        let attention_mask = vec![1; ids.len() + pair_ids.as_ref().map(|e| e.len()).unwrap_or(0)];

        Ok(Encoding::new(
            format!(
                "{}{}",
                encoding.get_original(),
                pair_encoding
                    .as_ref()
                    .map(|e| e.get_original())
                    .unwrap_or("")
            ),
            format!(
                "{}{}",
                encoding.get_normalized(),
                pair_encoding
                    .as_ref()
                    .map(|e| e.get_normalized())
                    .unwrap_or("")
            ),
            [&ids[..], &pair_ids.unwrap_or(vec![])[..]].concat(),
            [&type_ids[..], &pair_type_ids.unwrap_or(vec![])[..]].concat(),
            [&tokens[..], &pair_tokens.unwrap_or(vec![])[..]].concat(),
            [&offsets[..], &pair_offsets.unwrap_or(vec![])[..]].concat(),
            [
                &special_tokens[..],
                &pair_special_tokens.unwrap_or(vec![])[..],
            ]
            .concat(),
            attention_mask,
            encoding.take_overflowing(),
        ))
    }
}
