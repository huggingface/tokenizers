from .. import models

class Model:
    """Model
    """

    def save(folder: str, name: str) -> List[str]:
        """ save
        Save the current Model in the given folder, using the given name for the various
        files that will get created.
        Any file with the same name that already exist in this folder will be overwritten
        """
        pass

class BPE:
    """BPE
    """

    def from_files(vocab: str, merges: str) -> Model:
        pass

    def empty() -> Model:
        pass

class WordPiece:
    """WordPiece
    """

    def from_files(vocab: str) -> Model:
        pass

    def empty() -> Model:
        pass
