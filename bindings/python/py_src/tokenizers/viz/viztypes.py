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
    anno_ix: Optional[int]


class CharState():
    char_ix: Optional[int]

    def __init__(self,char_ix):
        self.char_ix = char_ix

        self.anno_ix: Optional[int] =None
        self.tokens: List[int] =[]
    @property
    def token_ix(self):
        return self.tokens[0] if len(self.tokens) >0 else None

    @property
    def is_multitoken(self):
        '''
        BPE tokenizers can output more than one token for a char
        '''
        return len(self.tokens) >1

    def partition_key(self) -> CharStateKey:
        return CharStateKey(token_ix=self.token_ix, anno_ix=self.anno_ix, )


class Aligned:
    pass
