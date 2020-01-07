__version__ = "0.0.12"

from .tokenizers import Tokenizer as TokenizerBackend, Encoding
from .tokenizers import decoders
from .tokenizers import models
from .tokenizers import normalizers
from .tokenizers import pre_tokenizers
from .tokenizers import processors
from .tokenizers import trainers

from typing import Optional, Union, List, Tuple


class Tokenizer(object):
    _tokenizer = None
    _decoder = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise NotImplementedError
        return self._tokenizer

    @property
    def decoder(self):
        if self._decoder is None:
            raise NotImplementedError
        return self._decoder

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=False)

    def __len__(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def _update_special_tokens(self):
        if self._tokenizer is not None:
            self._tokenizer.add_special_tokens(self.all_special_tokens)

    @staticmethod
    def _convert_encoding(
        encoding,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
    ):
        encoding_dict = {
            "input_ids": encoding.ids,
        }
        if return_token_type_ids:
            encoding_dict["token_type_ids"] = encoding.type_ids
        if return_attention_mask:
            encoding_dict["attention_mask"] = encoding.attention_mask
        if return_overflowing_tokens:
            overflowing = encoding.overflowing
            encoding_dict["overflowing_tokens"] = overflowing.ids if overflowing is not None else []
        if return_special_tokens_mask:
            encoding_dict["special_tokens_mask"] = encoding.special_tokens_mask

        # # Prepare inputs as tensors if asked
        # if return_tensors == "tf" and is_tf_available():
        #     encoding_dict["input_ids"] = tf.constant([encoding_dict["input_ids"]])
        #     encoding_dict["token_type_ids"] = tf.constant([encoding_dict["token_type_ids"]])

        #     if "attention_mask" in encoding_dict:
        #         encoding_dict["attention_mask"] = tf.constant([encoding_dict["attention_mask"]])

        # elif return_tensors == "pt" and is_torch_available():
        #     encoding_dict["input_ids"] = torch.tensor([encoding_dict["input_ids"]])
        #     encoding_dict["token_type_ids"] = torch.tensor([encoding_dict["token_type_ids"]])

        #     if "attention_mask" in encoding_dict:
        #         encoding_dict["attention_mask"] = torch.tensor([encoding_dict["attention_mask"]])
        # elif return_tensors is not None:
        #     logger.warning(
        #         "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
        #             return_tensors
        #         )
        #     )

        return encoding_dict

    def encode_plus(
        self,
        text,
        text_pair=None,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        **kwargs
    ):
        encoding = self.tokenizer.encode(text, text_pair)
        return self._convert_encoding(
            encoding,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        **kwargs
    ):
        """
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            stride=stride,
            truncation_strategy=truncation_strategy,
            pad_to_max_length=pad_to_max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id_with_added_voc(self, token):
        id = self.tokenizer.token_to_id(token)
        if id is None:
            return self.unk_token_id
        return id

    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_token(int(index))

    def convert_tokens_to_string(self, tokens):
        return self.decoder.decode(tokens)

    def add_tokens(self, new_tokens):
        """ Add the given tokens to the vocabulary

        Args:
            tokens: List[Union[str, Tuple[str, bool]]]:
                A list of tokens to add to the vocabulary. Each token can either be
                a string, or a tuple with a string representing the token, and a boolean
                option representing whether to match on single words only.
                If the boolean is not included, it defaults to False

        Returns:
            The number of tokens that were added to the vocabulary
        """
        self.tokenizer.add_tokens(new_tokens)

    def add_special_tokens(self, special_tokens_dict):
        """ Add the given special tokens to the vocabulary, and treat them as special tokens.

        The special tokens will never be processed by the model, and will be
        removed while decoding.

        Args:
            tokens: List[str]:
                The list of special tokens to add

        Returns:
            The number of tokens that were added to the vocabulary
        """
        added = super().add_special_tokens(special_tokens_dict)
        self._update_special_tokens()
        return added

    def encode_batch(
        self,
        texts,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
    ):
        """ Encode the given sequences or pair of sequences

        Args:
            sequences: List[Union[str, Tuple[str, str]]]:
                A list of sequences or pair of sequences. The list can contain both
                at the same time.

        Returns:
            A list of Encoding
        """
        return [
            self._convert_encoding(
                encoding,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
            )
            for encoding in self.tokenizer.encode_batch(texts)
        ]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """ Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def decode_batch(self, ids_batch, skip_special_tokens=False, clear_up_tokenization_spaces=True):
        """ Decode the list of sequences to a list of string sequences

        Args:
            sequences: List[List[unsigned int]]:
                A list of sequence of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output strings

        Returns:
            A list of decoded strings
        """
        return [
            self.clean_up_tokenization(text) if clear_up_tokenization_spaces else text
            for text in self.tokenizer.decode_batch(ids_batch, skip_special_tokens)
        ]

    def token_to_id(self, token: str) -> Optional[int]:
        """ Convert the given token to its corresponding id

        Args:
            token: str:
                The token to convert

        Returns:
            The corresponding id if it exists, None otherwise
        """
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """ Convert the given token id to its corresponding string

        Args:
            token: id:
                The token id to convert

        Returns:
            The corresponding string if it exists, None otherwise
        """
        return self._tokenizer.id_to_token(id)


class BPETokenizer(Tokenizer):
    """ BPE Tokenizer

    A Tokenizer works as a pipeline, it processes some raw text as input and outputs
    an `Encoding`.

    The various steps of the pipeline are:
        1. The `Normalizer`: in charge of normalizing the text. Common examples of
           normalization are the unicode normalization standards, such as NFD or NFKC.
        2. The `PreTokenizer`: in charge of creating initial words splits in the text.
           The most common way of splitting text is simply on whitespace.
        3. The `Model`: in charge of doing the actual tokenization. An example of a
           `Model` would be `BPE` or `WordPiece`.
        4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything
           relevant that, for example, a language model would need, such as special tokens.
    """
    def __init__(self, vocab_file=None,
                 merges_file=None,
                 byte_level=False,
                 add_prefix_space=False,
                 pad_to_max_length=False,
                 max_length=None,
                 stride=0,
                 truncation_strategy="longest_first"):
        """ Instantiate a new Tokenizer using the given Model

        Args:
            model: models.Model:
                The model to be used with this Tokenizer

        Returns:
            Tokenizer
        """
        self.byte_level = byte_level

        if vocab_file is not None and merges_file is not None:
            self._tokenizer = TokenizerBackend(models.BPE.from_files(vocab_file, merges_file))
        else:
            self._tokenizer = TokenizerBackend(models.BPE.empty())

        if byte_level:
            self._tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(add_prefix_space=add_prefix_space))
            self._tokenizer.with_decoder(decoders.ByteLevel.new())
            self._decoder = decoders.ByteLevel.new()
        else:
            self._tokenizer.with_pre_tokenizer(pre_tokenizers.BertPreTokenizer.new())
            self._tokenizer.with_decoder(decoders.ByteLevel.new())
            self._decoder = decoders.ByteLevel.new()

        if max_length:
            self._tokenizer.with_truncation(max_length, stride=stride, strategy=truncation_strategy)
            self._tokenizer.with_padding(
                max_length=max_length if pad_to_max_length else None,
                direction=self.padding_side,
                pad_id=self.pad_token_id if self.pad_token_id is not None else 0,
                pad_type_id=self.pad_token_type_id,
                pad_token=self.pad_token if self.pad_token is not None else "",
            )

    def train(self, files, vocab_size=50000, min_frequency=2, show_progress=True,
              special_tokens=["<s>", "<pad>", "</s>"]):
        if self.byte_level:
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
        else:
            initial_alphabet = []
        trainer = trainers.BpeTrainer.new(
                        vocab_size=vocab_size,
                        min_frequency=min_frequency,
                        show_progress=show_progress,
                        special_tokens=special_tokens,
                        initial_alphabet=initial_alphabet
                    )
        self._tokenizer.train(trainer, files)
