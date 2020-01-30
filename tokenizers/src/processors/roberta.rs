use crate::tokenizer::{Encoding, PostProcessor, Result};

pub struct RobertaProcessing {
    sep: (String, u32),
    cls: (String, u32),
}

impl RobertaProcessing {
    pub fn new(sep: (String, u32), cls: (String, u32)) -> Self {
        RobertaProcessing { sep, cls }
    }
}

impl PostProcessor for RobertaProcessing {
    fn added_tokens(
        &self,
        _encoding: &Encoding,
        pair_encoding: &Option<Encoding>,
    ) -> Result<usize> {
        if pair_encoding.is_some() {
            Ok(4)
        } else {
            Ok(2)
        }
    }

    fn process(&self, mut encoding: Encoding, pair_encoding: Option<Encoding>) -> Result<Encoding> {
        let ids = [&[self.cls.1], &encoding.get_ids()[..], &[self.sep.1]].concat();
        let type_ids = [&[0], &encoding.get_type_ids()[..], &[0]].concat();
        let tokens = [
            &[self.cls.0.clone()],
            &encoding.get_tokens()[..],
            &[self.sep.0.clone()],
        ]
        .concat();
        let offsets = [&[(0, 0)], &encoding.get_offsets()[..], &[(0, 0)]].concat();
        let special_tokens = [&[1u32], &vec![0; encoding.get_ids().len()][..], &[1]].concat();
        let attention_mask = vec![1; ids.len()];

        let mut new_encoding = Encoding::new(
            encoding.get_normalized().clone(),
            ids,
            type_ids,
            tokens,
            offsets,
            special_tokens,
            attention_mask,
            encoding.take_overflowing(),
        );

        if let Some(mut encoding) = pair_encoding {
            let pair_ids = [&[self.sep.1], &encoding.get_ids()[..], &[self.sep.1]].concat();
            let pair_type_ids = vec![0; encoding.get_ids().len() + 2];
            let pair_tokens = [
                &[self.sep.0.clone()],
                &encoding.get_tokens()[..],
                &[self.sep.0.clone()],
            ]
            .concat();
            let pair_offsets = [&[(0, 0)], &encoding.get_offsets()[..], &[(0, 0)]].concat();
            let pair_special_tokens =
                [&[1], &vec![0u32; encoding.get_type_ids().len()][..], &[1]].concat();
            let pair_attention_mask = vec![1; pair_ids.len()];

            let new_pair_encoding = Encoding::new(
                encoding.get_normalized().clone(),
                pair_ids,
                pair_type_ids,
                pair_tokens,
                pair_offsets,
                pair_special_tokens,
                pair_attention_mask,
                encoding.take_overflowing(),
            );

            new_encoding.merge_with(new_pair_encoding);
        }

        Ok(new_encoding)
    }
}
