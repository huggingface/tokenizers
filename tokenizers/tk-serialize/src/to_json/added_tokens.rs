use super::writer::Out;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

/// Writes the tokens in the order given: loading assigns the ids again in that order.
pub(super) fn write_added_tokens<'token>(
    out: &mut Out,
    tokens: impl IntoIterator<Item = (u32, &'token AddedToken)>,
) {
    out.arr_open();
    for (id, token) in tokens {
        out.obj_open();
        out.field_u32("id", id);
        write_added_token_fields(out, token);
        out.obj_close();
    }
    out.arr_close();
}

/// The keys of one `added_tokens` entry, except `id`.
pub(super) fn write_added_token_fields(out: &mut Out, token: &AddedToken) {
    out.field_str("content", &token.content);
    out.field_bool("single_word", token.single_word);
    out.field_bool("lstrip", token.lstrip);
    out.field_bool("rstrip", token.rstrip);
    out.field_bool("normalized", token.normalized);
    out.field_bool("special", token.special);
}
