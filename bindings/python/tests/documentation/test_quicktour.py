from ..utils import data_dir, doc_wiki_tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

disable_printing = True
original_print = print


def print(*args, **kwargs):
    if not disable_printing:
        original_print(*args, **kwargs)


class TestQuicktour:
    # This method contains everything we don't want to run
    @staticmethod
    def slow_train():
        tokenizer, trainer = TestQuicktour.get_tokenizer_trainer()

        # START train
        files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
        tokenizer.train(files, trainer)
        # END train
        # START save
        tokenizer.save("data/tokenizer-wiki.json")
        # END save

    @staticmethod
    def get_tokenizer_trainer():
        # START init_tokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import BPE

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        # END init_tokenizer
        # START init_trainer
        from tokenizers.trainers import BpeTrainer

        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        # END init_trainer
        # START init_pretok
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer.pre_tokenizer = Whitespace()
        # END init_pretok
        return tokenizer, trainer

    def test_quicktour(self, doc_wiki_tokenizer):
        def print(*args, **kwargs):
            pass

        try:
            # START reload_tokenizer
            tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
            # END reload_tokenizer
        except Exception:
            tokenizer = Tokenizer.from_file(doc_wiki_tokenizer)
        # START encode
        output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
        # END encode
        # START print_tokens
        print(output.tokens)
        # ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
        # END print_tokens
        assert output.tokens == [
            "Hello",
            ",",
            "y",
            "'",
            "all",
            "!",
            "How",
            "are",
            "you",
            "[UNK]",
            "?",
        ]
        # START print_ids
        print(output.ids)
        # [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
        # END print_ids
        assert output.ids == [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
        # START print_offsets
        print(output.offsets[9])
        # (26, 27)
        # END print_offsets
        assert output.offsets[9] == (26, 27)
        # START use_offsets
        sentence = "Hello, y'all! How are you 😁 ?"
        sentence[26:27]
        # "😁"
        # END use_offsets
        assert sentence[26:27] == "😁"
        # START check_sep
        tokenizer.token_to_id("[SEP]")
        # 2
        # END check_sep
        assert tokenizer.token_to_id("[SEP]") == 2
        # START init_template_processing
        from tokenizers.processors import TemplateProcessing

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )
        # END init_template_processing
        # START print_special_tokens
        output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
        print(output.tokens)
        # ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
        # END print_special_tokens
        assert output.tokens == [
            "[CLS]",
            "Hello",
            ",",
            "y",
            "'",
            "all",
            "!",
            "How",
            "are",
            "you",
            "[UNK]",
            "?",
            "[SEP]",
        ]
        # START print_special_tokens_pair
        output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
        print(output.tokens)
        # ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]
        # END print_special_tokens_pair
        assert output.tokens == [
            "[CLS]",
            "Hello",
            ",",
            "y",
            "'",
            "all",
            "!",
            "[SEP]",
            "How",
            "are",
            "you",
            "[UNK]",
            "?",
            "[SEP]",
        ]
        # START print_type_ids
        print(output.type_ids)
        # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        # END print_type_ids
        assert output.type_ids == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        # START encode_batch
        output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
        # END encode_batch
        # START encode_batch_pair
        output = tokenizer.encode_batch(
            [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
        )
        # END encode_batch_pair
        # START enable_padding
        tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
        # END enable_padding
        # START print_batch_tokens
        output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
        print(output[1].tokens)
        # ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
        # END print_batch_tokens
        assert output[1].tokens == ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
        # START print_attention_mask
        print(output[1].attention_mask)
        # [1, 1, 1, 1, 1, 1, 1, 0]
        # END print_attention_mask
        assert output[1].attention_mask == [1, 1, 1, 1, 1, 1, 1, 0]


if __name__ == "__main__":
    from urllib import request
    from zipfile import ZipFile
    import os

    disable_printing = False
    if not os.path.isdir("data/wikitext-103-raw"):
        print("Downloading wikitext-103...")
        wiki_text, _ = request.urlretrieve(
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
        )
        with ZipFile(wiki_text, "r") as z:
            print("Unzipping in data...")
            z.extractall("data")

    print("Now training...")
    TestQuicktour.slow_train()
