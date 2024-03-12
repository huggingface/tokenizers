from tokenizers import CharBPETokenizer

from ..utils import data_dir, multiprocessing_with_parallelism, openai_files


class TestCharBPETokenizer:
    def test_basic_encode(self, openai_files):
        tokenizer = CharBPETokenizer.from_file(openai_files["vocab"], openai_files["merges"])

        output = tokenizer.encode("My name is John", "pair")
        assert output.ids == [0, 253, 1362, 544, 0, 7, 12662, 2688]
        assert output.tokens == [
            "<unk>",
            "y</w>",
            "name</w>",
            "is</w>",
            "<unk>",
            "o",
            "hn</w>",
            "pair</w>",
        ]
        assert output.offsets == [
            (0, 1),
            (1, 2),
            (3, 7),
            (8, 10),
            (11, 12),
            (12, 13),
            (13, 15),
            (0, 4),
        ]
        assert output.type_ids == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_lowercase(self, openai_files):
        tokenizer = CharBPETokenizer.from_file(openai_files["vocab"], openai_files["merges"], lowercase=True)
        output = tokenizer.encode("My name is John", "pair", add_special_tokens=False)
        assert output.ids == [547, 1362, 544, 2476, 2688]
        assert output.tokens == ["my</w>", "name</w>", "is</w>", "john</w>", "pair</w>"]
        assert output.offsets == [(0, 2), (3, 7), (8, 10), (11, 15), (0, 4)]
        assert output.type_ids == [0, 0, 0, 0, 1]

    def test_decoding(self, openai_files):
        tokenizer = CharBPETokenizer.from_file(openai_files["vocab"], openai_files["merges"], lowercase=True)
        decoded = tokenizer.decode(tokenizer.encode("my name is john").ids)
        assert decoded == "my name is john"

    def test_multiprocessing_with_parallelism(self, openai_files):
        tokenizer = CharBPETokenizer.from_file(openai_files["vocab"], openai_files["merges"])
        multiprocessing_with_parallelism(tokenizer, False)
        multiprocessing_with_parallelism(tokenizer, True)

    def test_train_from_iterator(self):
        text = ["A first sentence", "Another sentence", "And a last one"]
        tokenizer = CharBPETokenizer()
        tokenizer.train_from_iterator(text, show_progress=False)

        output = tokenizer.encode("A sentence")
        assert output.tokens == ["A</w>", "sentence</w>"]
