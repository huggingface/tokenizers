from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from tokenizers.implementations import BaseTokenizer


class TestBaseTokenizer:
    def test_get_set_components(self):
        toki = Tokenizer(models.BPE())
        toki.normalizer = normalizers.NFC()
        toki.pre_tokenizer = pre_tokenizers.ByteLevel()
        toki.post_processor = processors.BertProcessing(("A", 0), ("B", 1))
        toki.decoder = decoders.ByteLevel()

        tokenizer = BaseTokenizer(toki)

        assert isinstance(tokenizer.model, models.BPE)
        assert isinstance(tokenizer.normalizer, normalizers.NFC)
        assert isinstance(tokenizer.pre_tokenizer, pre_tokenizers.ByteLevel)
        assert isinstance(tokenizer.post_processor, processors.BertProcessing)
        assert isinstance(tokenizer.decoder, decoders.ByteLevel)

        tokenizer.model = models.Unigram()
        assert isinstance(tokenizer.model, models.Unigram)
        tokenizer.normalizer = normalizers.NFD()
        assert isinstance(tokenizer.normalizer, normalizers.NFD)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        assert isinstance(tokenizer.pre_tokenizer, pre_tokenizers.Whitespace)
        tokenizer.post_processor = processors.ByteLevel()
        assert isinstance(tokenizer.post_processor, processors.ByteLevel)
        tokenizer.decoder = decoders.WordPiece()
        assert isinstance(tokenizer.decoder, decoders.WordPiece)
