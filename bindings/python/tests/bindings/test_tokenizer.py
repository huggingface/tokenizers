from tokenizers import AddedToken, Tokenizer
from tokenizers.models import Model, BPE


class TestAddedToken:
    def test_instantiate_with_content_only(self):
        added_token = AddedToken("<mask>")
        assert type(added_token) == AddedToken
        assert str(added_token) == "<mask>"
        assert (
            repr(added_token)
            == 'AddedToken("<mask>", rstrip=false, lstrip=false, single_word=false)'
        )
        assert added_token.rstrip == False
        assert added_token.lstrip == False
        assert added_token.single_word == False

    def test_can_set_rstrip(self):
        added_token = AddedToken("<mask>", rstrip=True)
        assert added_token.rstrip == True
        assert added_token.lstrip == False
        assert added_token.single_word == False

    def test_can_set_lstrip(self):
        added_token = AddedToken("<mask>", lstrip=True)
        assert added_token.rstrip == False
        assert added_token.lstrip == True
        assert added_token.single_word == False

    def test_can_set_single_world(self):
        added_token = AddedToken("<mask>", single_word=True)
        assert added_token.rstrip == False
        assert added_token.lstrip == False
        assert added_token.single_word == True


class TestTokenizer:
    def test_has_expected_type_and_methods(self):
        tokenizer = Tokenizer(BPE.empty())
        assert type(tokenizer) == Tokenizer
        assert callable(tokenizer.num_special_tokens_to_add)
        assert callable(tokenizer.get_vocab)
        assert callable(tokenizer.get_vocab_size)
        assert callable(tokenizer.enable_truncation)
        assert callable(tokenizer.no_truncation)
        assert callable(tokenizer.enable_padding)
        assert callable(tokenizer.no_padding)
        assert callable(tokenizer.normalize)
        assert callable(tokenizer.encode)
        assert callable(tokenizer.encode_batch)
        assert callable(tokenizer.decode)
        assert callable(tokenizer.decode_batch)
        assert callable(tokenizer.token_to_id)
        assert callable(tokenizer.id_to_token)
        assert callable(tokenizer.add_tokens)
        assert callable(tokenizer.add_special_tokens)
        assert callable(tokenizer.train)
        assert callable(tokenizer.post_process)
        assert isinstance(tokenizer.model, Model)
        assert tokenizer.normalizer is None
        assert tokenizer.pre_tokenizer is None
        assert tokenizer.post_processor is None
        assert tokenizer.decoder is None
