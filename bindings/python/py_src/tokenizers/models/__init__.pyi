# Generated content DO NOT EDIT
class Model:
    """
    A Model represents some tokenization algorithm like BPE or Word
    This class cannot be constructed directly. Please use one of the concrete models.
    """

    def id_to_token(self, id):
        """
        Returns the token associated with the given id
        """
        pass
    def save(self, folder, name):
        """
        Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass
    def token_to_id(self, tokens):
        """
        Returns the id associated with the given token
        """
        pass
    def tokenize(self, tokens):
        """
        Tokenize the given sequence
        """
        pass

class BPE(Model):
    """
    Instantiate a BPE Model from the given vocab and merges.

    Args:
       vocab: ('`optional`) Dict[str, int]:
           A dictionnary of string keys and their ids {"am": 0,...}

       merges: (`optional`) string:
           A list of pairs of tokens [("a", "b"),...]

       cache_capacity: (`optional`) int:
           The number of words that the BPE cache can contain. The cache allows
           to speed-up the process by keeping the result of the merge operations
           for a number of words.

       dropout: (`optional`) Optional[float] [0, 1]:
           The BPE dropout to use. Must be an float between 0 and 1

       unk_token: (`optional`) str:
           The unknown token to be used by the model.

       continuing_subword_prefix: (`optional`) str:
           The prefix to attach to subword units that don't represent a beginning of word.

       end_of_word_suffix: (`optional`) str:
           The suffix to attach to subword units that represent an end of word.

       fuse_unk: (`optional`) bool:
           Multiple unk tokens get fused into only 1
    """

    def __init__(
        self,
        vocab=None,
        merges=None,
        cache_capacity=None,
        dropout=None,
        unk_token=None,
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
        fuse_unk=None,
    ):
        pass
    @staticmethod
    def from_file(vocab_filename, merge_filename, **kwargs):
        """
        Convenient method to intialize a BPE from files
        Roughly equivalent to

        def from_file(vocab_filename, merges_filenames, **kwargs):
            vocab, merges = BPE.read_file(vocab_filename, merges_filename)
            return BPE(vocab, merges, **kwargs)
        """
        pass
    def id_to_token(self, id):
        """
        Returns the token associated with the given id
        """
        pass
    @staticmethod
    def read_file(self, vocab_filename, merges_filename):
        """
        Read a vocab_filename and merge_filename and stores result in memory
        """
        pass
    def save(self, folder, name):
        """
        Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass
    def token_to_id(self, tokens):
        """
        Returns the id associated with the given token
        """
        pass
    def tokenize(self, tokens):
        """
        Tokenize the given sequence
        """
        pass

class Unigram(Model):
    """
    UnigramEncoding model class

    Instantiate a Unigram Model from the given model file.

    Args:
       vocab: ('`optional`) string:
           A list of vocabulary items and their relative score [("am", -0.2442),...]

    """

    def __init__(self, vocab):
        pass
    def id_to_token(self, id):
        """
        Returns the token associated with the given id
        """
        pass
    def save(self, folder, name):
        """
        Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass
    def token_to_id(self, tokens):
        """
        Returns the id associated with the given token
        """
        pass
    def tokenize(self, tokens):
        """
        Tokenize the given sequence
        """
        pass

class WordLevel(Model):
    """
    Most simple tokenizer model based on mapping token from a vocab file to their corresponding id.

    Instantiate a WordLevel Model from the given vocab file.

        Args:
            vocab: (`optional`) string:
                A dictionnary of string keys and their ids {"am": 0,...}

            unk_token: str:
                The unknown token to be used by the model.
    """

    def __init__(self, vocab, unk_token):
        pass
    def id_to_token(self, id):
        """
        Returns the token associated with the given id
        """
        pass
    def save(self, folder, name):
        """
        Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass
    def token_to_id(self, tokens):
        """
        Returns the id associated with the given token
        """
        pass
    def tokenize(self, tokens):
        """
        Tokenize the given sequence
        """
        pass

class WordPiece(Model):
    """
    WordPiece model
    Instantiate a WordPiece Model from the given vocab file.

    Args:
        vocab: (`optional`) string:
            A dictionnary of string keys and their ids {"am": 0,...}

        unk_token: (`optional`) str:
            The unknown token to be used by the model.

        max_input_chars_per_word: (`optional`) int:
            The maximum number of characters to authorize in a single word.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word):
        pass
    @staticmethod
    def from_file(vocab_filename, merge_filename, **kwargs):
        """
        Convenient method to intialize a WordPiece from files
        Roughly equivalent to

        def from_file(vocab_filename, **kwargs):
            vocab = WordPiece.read_file(vocab_filename)
            return WordPiece(vocab, **kwargs)
        """
        pass
    def id_to_token(self, id):
        """
        Returns the token associated with the given id
        """
        pass
    @staticmethod
    def read_file(vocab_filename):
        """
        Read a vocab_filename and stores result in memory
        """
        pass
    def save(self, folder, name):
        """
        Save the current model

        Save the current model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten.
        """
        pass
    def token_to_id(self, tokens):
        """
        Returns the id associated with the given token
        """
        pass
    def tokenize(self, tokens):
        """
        Tokenize the given sequence
        """
        pass
