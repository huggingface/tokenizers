// Every fallible tk_* function must report a clean TkError instead of crashing when a
// required pointer argument is NULL. Every tk_*_free function must be a safe no-op on NULL,
// and safe to call twice on the same handle. This exercises every such case against the real
// dylib, the same way encode.c exercises the happy path.
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

static TkHandle_TkTokenizer load_tokenizer(const char *path)
{
    TkHandle_TkTokenizer tok = NULL;
    TkHandle_TkError err = tk_tokenizer_from_file(path, &tok);
    if (err != NULL)
    {
        (void)fprintf(stderr, "setup: tk_tokenizer_from_file failed: %s\n", tk_error_message(err));
        exit(2);
    }
    return tok;
}

static TkHandle_TkEncoding encode_hello(TkHandle_TkTokenizer tok)
{
    TkHandle_TkEncoding enc = NULL;
    const char *text = "Hello, world!";
    TkHandle_TkError err = tk_tokenizer_encode(tok, text, strlen(text), true, &enc);
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
    tk_error_free(NULL);
    printf("ok: tk_*_free(NULL) doesn't crash\n");

    {
        TkHandle_TkTokenizer tok = load_tokenizer(fixture);
        tk_tokenizer_free(&tok);
        tk_tokenizer_free(&tok);
        check(tok == NULL, "tk_tokenizer_free() is a no-op the second time on the same handle");
    }

    {
        TkHandle_TkTokenizer tok = NULL;
        TkHandle_TkError err = tk_tokenizer_from_file(NULL, &tok);
        check(err != NULL && tok == NULL, "tk_tokenizer_from_file(NULL path) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_TkError err = tk_tokenizer_from_file(fixture, NULL);
        check(err != NULL, "tk_tokenizer_from_file(..., NULL out) reports an error");
        tk_error_free(&err);
    }

    TkHandle_TkTokenizer tok = load_tokenizer(fixture);

    {
        TkHandle_TkEncoding enc = NULL;
        TkHandle_TkError err = tk_tokenizer_encode(NULL, "hi", 2, true, &enc);
        check(err != NULL && enc == NULL, "tk_tokenizer_encode(NULL tokenizer) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_TkEncoding enc = NULL;
        TkHandle_TkError err = tk_tokenizer_encode(tok, NULL, 5, true, &enc);
        check(err != NULL && enc == NULL,
              "tk_tokenizer_encode(NULL input, len=5) reports an error");
        tk_error_free(&err);
    }
    {
        // NULL input is rejected even at len=0: a TkSlice can't tell "no text" apart from
        // an actually-invalid pointer, so it's simplest and safest to just require non-NULL.
        TkHandle_TkEncoding enc = NULL;
        TkHandle_TkError err = tk_tokenizer_encode(tok, NULL, 0, true, &enc);
        check(err != NULL && enc == NULL,
              "tk_tokenizer_encode(NULL input, len=0) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_TkError err = tk_tokenizer_encode(tok, "hi", 2, true, NULL);
        check(err != NULL, "tk_tokenizer_encode(..., NULL out) reports an error");
        tk_error_free(&err);
    }

    TkHandle_TkEncoding enc = encode_hello(tok);

    {
        struct TkSlice_u32 out;
        TkHandle_TkError err = tk_encoding_ids(NULL, &out);
        check(err != NULL, "tk_encoding_ids(NULL encoding) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_TkError err = tk_encoding_ids(enc, NULL);
        check(err != NULL, "tk_encoding_ids(..., NULL out) reports an error");
        tk_error_free(&err);
    }
    {
        struct TkSlice_u8 out;
        TkHandle_TkError err = tk_encoding_type_ids(NULL, &out);
        check(err != NULL, "tk_encoding_type_ids(NULL encoding) reports an error");
        tk_error_free(&err);
    }
    {
        TkHandle_TkError err = tk_encoding_type_ids(enc, NULL);
        check(err != NULL, "tk_encoding_type_ids(..., NULL out) reports an error");
        tk_error_free(&err);
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
