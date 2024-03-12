import json
import os
import unittest

import tqdm

from huggingface_hub import HfApi, cached_download, hf_hub_url
from tokenizers import Tokenizer
from .utils import albert_base, data_dir


class TestSerialization:
    def test_full_serialization_albert(self, albert_base):
        # Check we can read this file.
        # This used to fail because of BufReader that would fail because the
        # file exceeds the buffer capacity
        Tokenizer.from_file(albert_base)


def check(tokenizer_file) -> bool:
    with open(tokenizer_file, "r") as f:
        data = json.load(f)
    if "pre_tokenizer" not in data:
        return True
    if "type" not in data["pre_tokenizer"]:
        return False
    if data["pre_tokenizer"]["type"] == "Sequence":
        for pre_tok in data["pre_tokenizer"]["pretokenizers"]:
            if "type" not in pre_tok:
                return False
    return True


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    if os.getenv("RUN_SLOW") != "1":
        return unittest.skip("use `RUN_SLOW=1` to run")(test_case)
    else:
        return test_case


@slow
class TestFullDeserialization(unittest.TestCase):
    def test_full_deserialization_hub(self):
        # Check we can read this file.
        # This used to fail because of BufReader that would fail because the
        # file exceeds the buffer capacity
        not_loadable = []
        invalid_pre_tokenizer = []

        # models = api.list_models(filter="transformers")
        # for model in tqdm.tqdm(models):
        #     model_id = model.modelId
        #     for model_file in model.siblings:
        #         filename = model_file.rfilename
        #         if filename == "tokenizer.json":
        #             all_models.append((model_id, filename))

        all_models = [("HueyNemud/das22-10-camembert_pretrained", "tokenizer.json")]
        for model_id, filename in tqdm.tqdm(all_models):
            tokenizer_file = cached_download(hf_hub_url(model_id, filename=filename))

            is_ok = check(tokenizer_file)
            if not is_ok:
                print(f"{model_id} is affected by no type")
                invalid_pre_tokenizer.append(model_id)
            try:
                Tokenizer.from_file(tokenizer_file)
            except Exception as e:
                print(f"{model_id} is not loadable: {e}")
                not_loadable.append(model_id)
            except:  # noqa: E722
                print(f"{model_id} is not loadable: Rust error")
                not_loadable.append(model_id)

            self.assertEqual(invalid_pre_tokenizer, [])
            self.assertEqual(not_loadable, [])
