//! Most important part is to save in order of ids!

use super::writer::Out;
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary;

pub(super) fn write_added_tokens(out: &mut Out, added: &BucketAddedVocabulary) {
    let mut tokens: Vec<_> = added.get_added_tokens_decoder().into_iter().collect();
    tokens.sort_unstable_by_key(|&(id, _)| id);

    out.arr_open();
    for (id, token) in &tokens {
        out.obj_open();
        out.field_u32("id", *id);
        out.field_str("content", &token.content);
        out.field_bool("single_word", token.single_word);
        out.field_bool("lstrip", token.lstrip);
        out.field_bool("rstrip", token.rstrip);
        out.field_bool("normalized", token.normalized);
        out.field_bool("special", token.special);
        out.obj_close();
    }
    out.arr_close();
}
