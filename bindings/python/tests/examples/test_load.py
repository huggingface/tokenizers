from tokenizers import Tokenizer

# START load
tokenizer = Tokenizer.from_file("data/roberta.json")
# END load

example = "This is an example"
ids = [713, 16, 41, 1246]
tokens = ["This", "Ġis", "Ġan", "Ġexample"]

encodings = tokenizer.encode(example)

assert encodings.ids == ids
assert encodings.tokens == tokens

decoded = tokenizer.decode(ids)
assert decoded == example
