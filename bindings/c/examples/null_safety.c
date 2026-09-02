// Every fallible tk_* function must report a clean TkError instead of crashing when a
// required pointer argument is NULL. Every tk_*_free function must be a safe no-op on NULL,
// and safe to call twice on the same pointer. This exercises every such case against the real
// dylib, the same way quick_start.c exercises the happy path.
//
// It also exercises TkEncodeOptions/TkDecodeOptions: NULL means defaults, a set option carries
// through to the encode/decode call, and every set rejects NULL options the same way every
// other tk_* function rejects a NULL pointer.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizers/tokenizers.h"

static int failures = 0;

static void check(int ok, const char *what)
{
    if (ok)
    {
        printf("ok: %s\n", what);
    }
    else
    {
        (void)fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static TkTokenizer *load_tokenizer(const char *path)
{
    TkTokenizer *tok = NULL;
    TkError *err = tk_tokenizer_from_file(path, &tok);
    if (err != NULL)
    {
        (void)fprintf(stderr, "setup: tk_tokenizer_from_file failed: %s\n", tk_error_message(err));
        exit(2);
    }
    return tok;
}

static TkEncoding *encode_hello(TkTokenizer *tok)
{
    TkEncoding *enc = NULL;
    const char *text = "Hello, world!";
    TkError *err = tk_tokenizer_encode(tok, text, strlen(text), NULL, &enc);
    if (err != NULL)
    {
        (void)fprintf(stderr, "setup: tk_tokenizer_encode failed: %s\n", tk_error_message(err));
        exit(2);
    }
    return enc;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        (void)fprintf(stderr, "usage: %s <tokenizer.json>\n", argv[0]);
        return 1;
    }
    const char *fixture = argv[1];

    check(tk_error_message(NULL) == NULL, "tk_error_message(NULL) returns NULL");

    tk_tokenizer_free(NULL);
    tk_encoding_free(NULL);
    tk_decoded_string_free(NULL);
    tk_error_free(NULL);
    tk_encode_options_free(NULL);
    tk_decode_options_free(NULL);
    printf("ok: tk_*_free(NULL) doesn't crash\n");

    {
        TkTokenizer *tok = load_tokenizer(fixture);
        tk_tokenizer_free(&tok);
        tk_tokenizer_free(&tok);
        check(tok == NULL, "tk_tokenizer_free() is a no-op the second time on the same pointer");
    }

    {
        TkTokenizer *tok = NULL;
        TkError *err = tk_tokenizer_from_file(NULL, &tok);
        check(err != NULL && tok == NULL, "tk_tokenizer_from_file(NULL path) reports an error");
        tk_error_free(&err);
    }
    {
        TkError *err = tk_tokenizer_from_file(fixture, NULL);
        check(err != NULL, "tk_tokenizer_from_file(..., NULL out) reports an error");
        tk_error_free(&err);
    }

    TkTokenizer *tok = load_tokenizer(fixture);

    {
        TkEncoding *enc = NULL;
        TkError *err = tk_tokenizer_encode(NULL, "hi", 2, NULL, &enc);
        check(err != NULL && enc == NULL, "tk_tokenizer_encode(NULL tokenizer) reports an error");
        tk_error_free(&err);
    }
    {
        TkEncoding *enc = NULL;
        TkError *err = tk_tokenizer_encode(tok, NULL, 5, NULL, &enc);
        check(err != NULL && enc == NULL,
              "tk_tokenizer_encode(NULL input, len=5) reports an error");
        tk_error_free(&err);
    }
    {
        // NULL input is rejected even at len=0: a TkSlice can't tell "no text" apart from
        // an actually-invalid pointer, so it's simplest and safest to just require non-NULL.
        TkEncoding *enc = NULL;
        TkError *err = tk_tokenizer_encode(tok, NULL, 0, NULL, &enc);
        check(err != NULL && enc == NULL,
              "tk_tokenizer_encode(NULL input, len=0) reports an error");
        tk_error_free(&err);
    }
    {
        TkError *err = tk_tokenizer_encode(tok, "hi", 2, NULL, NULL);
        check(err != NULL, "tk_tokenizer_encode(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkEncodeOptions *options = NULL;
        TkError *err = tk_encode_options_set_add_special_tokens(options, true);
        check(err != NULL, "tk_encode_options_set_add_special_tokens(NULL options) reports an error");
        tk_error_free(&err);
    }
    {
        TkError *err = tk_encode_options_new(NULL);
        check(err != NULL, "tk_encode_options_new(NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkEncodeOptions *options = NULL;
        TkError *err = tk_encode_options_new(&options);
        check(err == NULL && options != NULL, "tk_encode_options_new() succeeds");
        err = tk_encode_options_set_add_special_tokens(options, false);
        check(err == NULL, "tk_encode_options_set_add_special_tokens(options, false) succeeds");

        TkEncoding *enc = NULL;
        err = tk_tokenizer_encode(tok, "hi", 2, options, &enc);
        check(err == NULL && enc != NULL, "tk_tokenizer_encode(explicit options) succeeds");
        tk_encoding_free(&enc);

        tk_encode_options_free(&options);
        tk_encode_options_free(&options);
        check(options == NULL, "tk_encode_options_free() is a no-op the second time on the same pointer");
    }

    TkEncoding *enc = encode_hello(tok);

    {
        TkSlice_u32 out;
        TkError *err = tk_encoding_ids(NULL, &out);
        check(err != NULL, "tk_encoding_ids(NULL encoding) reports an error");
        tk_error_free(&err);
    }
    {
        TkError *err = tk_encoding_ids(enc, NULL);
        check(err != NULL, "tk_encoding_ids(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkSlice_u8 out;
        TkError *err = tk_encoding_type_ids(NULL, &out);
        check(err != NULL, "tk_encoding_type_ids(NULL encoding) reports an error");
        tk_error_free(&err);
    }
    {
        TkError *err = tk_encoding_type_ids(enc, NULL);
        check(err != NULL, "tk_encoding_type_ids(..., NULL out) reports an error");
        tk_error_free(&err);
    }

    {
        TkSlice_u32 ids = {0};
        TkDecodedString *dec = NULL;
        TkError *err = tk_tokenizer_decode(NULL, ids, NULL, &dec);
        check(err != NULL && dec == NULL, "tk_tokenizer_decode(NULL tokenizer) reports an error");
        tk_error_free(&err);
    }
    {
        TkSlice_u32 ids = {0};
        TkError *err = tk_tokenizer_decode(tok, ids, NULL, NULL);
        check(err != NULL, "tk_tokenizer_decode(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        // Unlike tk_tokenizer_encode's input, a NULL ids.ptr is valid as long as len is 0: a
        // TkSlice can't tell "no ids" apart from an actually-invalid pointer, and here "no ids"
        // is a legitimate, well-defined input (decodes to an empty string).
        TkSlice_u32 ids = {NULL, 0};
        TkDecodedString *dec = NULL;
        TkError *err = tk_tokenizer_decode(tok, ids, NULL, &dec);
        check(err == NULL && dec != NULL,
              "tk_tokenizer_decode(NULL ids.ptr, len=0) succeeds as an empty decode");
        tk_decoded_string_free(&dec);
    }
    {
        TkDecodeOptions *options = NULL;
        TkError *err = tk_decode_options_set_skip_special_tokens(options, true);
        check(err != NULL, "tk_decode_options_set_skip_special_tokens(NULL options) reports an error");
        tk_error_free(&err);
    }
    {
        TkError *err = tk_decode_options_new(NULL);
        check(err != NULL, "tk_decode_options_new(NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        TkDecodeOptions *options = NULL;
        TkError *err = tk_decode_options_new(&options);
        check(err == NULL && options != NULL, "tk_decode_options_new() succeeds");
        err = tk_decode_options_set_skip_special_tokens(options, false);
        check(err == NULL, "tk_decode_options_set_skip_special_tokens(options, false) succeeds");

        TkSlice_u32 ids = {NULL, 0};
        TkDecodedString *dec = NULL;
        err = tk_tokenizer_decode(tok, ids, options, &dec);
        check(err == NULL && dec != NULL, "tk_tokenizer_decode(explicit options) succeeds");
        tk_decoded_string_free(&dec);

        tk_decode_options_free(&options);
        tk_decode_options_free(&options);
        check(options == NULL, "tk_decode_options_free() is a no-op the second time on the same pointer");
    }

    {
        TkSlice_u32 ids = {NULL, 0};
        TkDecodedString *dec = NULL;
        TkError *err = tk_tokenizer_decode(tok, ids, NULL, &dec);
        if (err != NULL)
        {
            (void)fprintf(stderr, "setup: tk_tokenizer_decode failed: %s\n", tk_error_message(err));
            exit(2);
        }

        {
            TkSlice_u8 out;
            TkError *bytes_err = tk_decoded_string_bytes(NULL, &out);
            check(bytes_err != NULL, "tk_decoded_string_bytes(NULL decoded_string) reports an error");
            tk_error_free(&bytes_err);
        }
        {
            TkError *bytes_err = tk_decoded_string_bytes(dec, NULL);
            check(bytes_err != NULL, "tk_decoded_string_bytes(..., NULL out) reports an error");
            tk_error_free(&bytes_err);
        }

        tk_decoded_string_free(&dec);
        tk_decoded_string_free(&dec);
        check(dec == NULL, "tk_decoded_string_free() is a no-op the second time on the same pointer");
    }

    tk_encoding_free(&enc);
    tk_tokenizer_free(&tok);

    if (failures > 0)
    {
        (void)fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all null-safety checks passed\n");
    return 0;
}
