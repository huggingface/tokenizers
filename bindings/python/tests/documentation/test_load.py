from tokenizers import Tokenizer


def test_load_tokenizer():
    # START load_tokenizer
    tokenizer = Tokenizer.from_file("data/roberta.json")
    # END load_tokenizer

    example = "This is an example"
    ids = [713, 16, 41, 1246]
    tokens = ["This", "Ġis", "Ġan", "Ġexample"]

    encodings = tokenizer.encode(example)

    assert encodings.ids == ids
    assert encodings.tokens == tokens

    decoded = tokenizer.decode(ids)
    assert decoded == example
