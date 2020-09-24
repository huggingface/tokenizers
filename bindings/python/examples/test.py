# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import ByteLevel
# from tokenizers.normalizers import NFKC, NFC, Lowercase, Sequence
#
# tok = Tokenizer(BPE("../../data/roberta-base-vocab.json", "../../data/roberta-base-merges.txt"))
# tok.pre_tokenizer = ByteLevel()
# tok.normalizer = Sequence([NFC(), NFKC()])
#
# tok.save("THE_TEST.tokenizer.json", pretty=True)
# print(tok.encode("𝕿𝖍𝖊 𝖖𝖚𝖎𝖈𝖐, 𝖇𝖗𝖔𝖜𝖓 🦊 𝖏𝖚𝖒𝖕𝖘 𝖔𝖛𝖊𝖗 𝖙𝖍𝖊 𝖑𝖆𝖟𝖞 🐶").tokens)
#
# tok = Tokenizer.from_file("THE_TEST.tokenizer.json")
# # with open("THE_TEST.tokenizer.json", "r") as f:
# #     t = f.read()
# #     tok = Tokenizer.from_str(t)
# print(tok.encode("𝕿𝖍𝖊 𝖖𝖚𝖎𝖈𝖐, 𝖇𝖗𝖔𝖜𝖓 🦊 𝖏𝖚𝖒𝖕𝖘 𝖔𝖛𝖊𝖗 𝖙𝖍𝖊 𝖑𝖆𝖟𝖞 🐶").tokens)

from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast, LineByLineTextDataset

# tokenizer = Tokenizer(
#     BPE("../../data/roberta-base-vocab.json", "../../data/roberta-base-merges.txt")
# )
tokenizer = Tokenizer.from_file("../../data/roberta-tok.tokenizer")
print(tokenizer.encode("Hello there!").tokens)

tok_transformers = PreTrainedTokenizerFast(BaseTokenizer(tokenizer))
print(tok_transformers.tokenize("Hello there!"))

dataset = LineByLineTextDataset(tokenizer=tok_transformers, file_path="../../data/botchan.txt", block_size=12)


# tokenizer = ByteLevelBPETokenizer.from_files(
#     "../../data/roberta-base-vocab.json", "../../data/roberta-base-merges.txt"
# )
# print(tokenizer.encode("Hello there!").tokens)
