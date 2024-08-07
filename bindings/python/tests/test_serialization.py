import json
import os
import unittest

import tqdm
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram

from .utils import albert_base, data_dir


class TestSerialization:
    def test_full_serialization_albert(self, albert_base):
        # Check we can read this file.
        # This used to fail because of BufReader that would fail because the
        # file exceeds the buffer capacity
        Tokenizer.from_file(albert_base)

    def test_str_big(self, albert_base):
        tokenizer = Tokenizer.from_file(albert_base)
        assert (
            str(tokenizer)
            == """Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[{"id":0, "content":"<pad>", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":1, "content":"<unk>", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":2, "content":"[CLS]", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":3, "content":"[SEP]", "single_word":False, "lstrip":False, "rstrip":False, ...}, {"id":4, "content":"[MASK]", "single_word":False, "lstrip":False, "rstrip":False, ...}], normalizer=Sequence(normalizers=[Replace(pattern=String("``"), content="\""), Replace(pattern=String("''"), content="\""), NFKD(), StripAccents(), Lowercase(), ...]), pre_tokenizer=Sequence(pretokenizers=[WhitespaceSplit(), Metaspace(replacement="▁", prepend_scheme=always, split=True)]), post_processor=TemplateProcessing(single=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0)], pair=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0), Sequence(id=B, type_id=1), SpecialToken(id="[SEP]", type_id=1)], special_tokens={"[CLS]":SpecialToken(id="[CLS]", ids=[2], tokens=["[CLS]"]), "[SEP]":SpecialToken(id="[SEP]", ids=[3], tokens=["[SEP]"])}), decoder=Metaspace(replacement="▁", prepend_scheme=always, split=True), model=Unigram(unk_id=1, vocab=[("<pad>", 0), ("<unk>", 0), ("[CLS]", 0), ("[SEP]", 0), ("[MASK]", 0), ...], byte_fallback=False))"""
        )

    def test_repr_str(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my"])
        assert (
            repr(tokenizer)
            == """Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[{"id":0, "content":"my", "single_word":False, "lstrip":False, "rstrip":False, "normalized":True, "special":False}], normalizer=None, pre_tokenizer=None, post_processor=None, decoder=None, model=BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[]))"""
        )
        assert (
            str(tokenizer)
            == """Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[{"id":0, "content":"my", "single_word":False, "lstrip":False, "rstrip":False, ...}], normalizer=None, pre_tokenizer=None, post_processor=None, decoder=None, model=BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[]))"""
        )

    def test_repr_str_ellipsis(self):
        model = BPE()
        assert (
            repr(model)
            == """BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[])"""
        )
        assert (
            str(model)
            == """BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[])"""
        )

        vocab = [
            ("A", 0.0),
            ("B", -0.01),
            ("C", -0.02),
            ("D", -0.03),
            ("E", -0.04),
        ]
        # No ellispsis yet
        model = Unigram(vocab, 0, byte_fallback=False)
        assert (
            repr(model)
            == """Unigram(unk_id=0, vocab=[("A", 0), ("B", -0.01), ("C", -0.02), ("D", -0.03), ("E", -0.04)], byte_fallback=False)"""
        )
        assert (
            str(model)
            == """Unigram(unk_id=0, vocab=[("A", 0), ("B", -0.01), ("C", -0.02), ("D", -0.03), ("E", -0.04)], byte_fallback=False)"""
        )

        # Ellispis for longer than 5 elements only on `str`.
        vocab = [
            ("A", 0.0),
            ("B", -0.01),
            ("C", -0.02),
            ("D", -0.03),
            ("E", -0.04),
            ("F", -0.04),
        ]
        model = Unigram(vocab, 0, byte_fallback=False)
        assert (
            repr(model)
            == """Unigram(unk_id=0, vocab=[("A", 0), ("B", -0.01), ("C", -0.02), ("D", -0.03), ("E", -0.04), ("F", -0.04)], byte_fallback=False)"""
        )
        assert (
            str(model)
            == """Unigram(unk_id=0, vocab=[("A", 0), ("B", -0.01), ("C", -0.02), ("D", -0.03), ("E", -0.04), ...], byte_fallback=False)"""
        )


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
            tokenizer_file = hf_hub_download(model_id, filename=filename)

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
