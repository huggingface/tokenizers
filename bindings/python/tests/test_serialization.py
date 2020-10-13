from tokenizers import Tokenizer, models, normalizers
from .utils import data_dir, precompiled_files


class TestSerialization:
    def test_full_serialization_albert(self, precompiled_files):
        # Check we can read this file.
        # This used to fail because of BufReader that would fail because the
        # file exceeds the buffer capacity
        tokenizer = Tokenizer.from_file(precompiled_files["albert_base"])
