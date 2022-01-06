from tokenizers import Tokenizer, pre_tokenizers, processors


tokenizer = Tokenizer.from_pretrained("roberta-base")

add_prefix_space = True
trim_offsets = True
string = "the the"

encoded = tokenizer.encode(string)
print(encoded.ids, encoded.offsets)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=add_prefix_space, trim_offsets=trim_offsets
)
tokenizer.post_processor = processors.ByteLevel(
    add_prefix_space=add_prefix_space, trim_offsets=trim_offsets
)

encoded = tokenizer.encode(string)
print(encoded.ids, encoded.offsets)

import ipdb

ipdb.set_trace()
