from typing import List, Optional, Tuple, NamedTuple


class Annotation:
    start: int
    end: int
    label: int

    def __init__(self, start: int, end: int, label: str):
        self.start = start
        self.end = end
        self.label = label


AnnotationList = List[Annotation]
PartialIntList = List[Optional[int]]


class CharStateKey(NamedTuple):
    token_ix: Optional[int]
    word_ix: Optional[int]
    anno_ix: Optional[int]


class CharState(NamedTuple):
    char_ix: Optional[int]
    token_ix: Optional[int]
    word_ix: Optional[int]
    anno_ix: Optional[int]

    def partition_key(self) -> CharStateKey:
        return CharStateKey(token_ix=self.token_ix, anno_ix=self.anno_ix, word_ix=self.word_ix)


class Aligned:
    pass
