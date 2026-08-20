//! The `added_tokens` array.
//!
//! Sorted by id ascending, which is not cosmetic: the reader replays these into the vocabulary in id
//! order, because `add_tokens` reuses a model id when the token is already there. Writing them in
//! any other order would produce a file that reads back as a different set of ids.

use super::writer::Out;
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary;

pub(super) fn write_added_tokens(out: &mut Out, added: &BucketAddedVocabulary) {
    // The decoder is a map, so its iteration order is not the id order the reader needs.
    let mut tokens: Vec<_> = added.get_added_tokens_decoder().into_iter().collect();
    tokens.sort_unstable_by_key(|&(id, _)| id);

    out.arr_open();
    for (id, token) in &tokens {
        out.obj_open();
        out.field_u32("id", *id);
        out.field_str("content", &token.content);
        // All six flags, always. The reader requires every one of them -- as does the config path,
        // where `AddedToken` has no serde defaults -- so an omitted flag is a file neither can load.
        out.field_bool("single_word", token.single_word);
        out.field_bool("lstrip", token.lstrip);
        out.field_bool("rstrip", token.rstrip);
        out.field_bool("normalized", token.normalized);
        out.field_bool("special", token.special);
        out.obj_close();
    }
    out.arr_close();
}
