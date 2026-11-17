import subprocess
import sys

from conftest import GPT2

# A subprocess, since a test cannot unload the numpy this interpreter already imported.
# `sys.modules["numpy"] = None` makes `import numpy` raise, as on an install without it.
WITHOUT_NUMPY = f"""
import sys
sys.modules["numpy"] = None

from tokenizers import Tokenizer

gpt2 = Tokenizer.from_file({str(GPT2)!r}).compile()
encoding = gpt2.encode("Hello there")
assert gpt2.decode(encoding.ids) == gpt2.decode(encoding) == "Hello there"
assert encoding.ids_buffer.tolist() == encoding.ids
"""


def test_encode_and_decode_without_numpy():
    subprocess.run([sys.executable, "-c", WITHOUT_NUMPY], check=True)
