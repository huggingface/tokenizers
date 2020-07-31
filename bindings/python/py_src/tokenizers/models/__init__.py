from typing import List, Tuple

from .. import models, Offsets

TokenizedSequence = List[str]
TokenizedSequenceWithOffsets = List[Tuple[str, Offsets]]

Model = models.Model
BPE = models.BPE
WordPiece = models.WordPiece
WordLevel = models.WordLevel
