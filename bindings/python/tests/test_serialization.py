from tokenizers import Tokenizer, models, normalizers
from .utils import data_dir, albert_base


class TestSerialization:
    def test_full_serialization_albert(self, albert_base):
        # Check we can read this file.
        # This used to fail because of BufReader that would fail because the
        # file exceeds the buffer capacity
        tokenizer = Tokenizer.from_file(albert_base)
