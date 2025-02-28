from abc import ABC, abstractmethod
from typing import Iterable, List


class Tokenizer(ABC):
    "Separates sentences into lists of tokens. Can be either a static tokenizer or a trainable one"
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: Iterable[str]) -> None:
        pass

    @abstractmethod
    def transform(self, data: Iterable[str]) -> List[List[str]]:
        pass

    @abstractmethod
    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        pass

from .bigram_tokenizer import BigramTokenizer
from .char_tokenizer import CharTokenizer
from .short_word_tokenizer import ShortWordTokenizer
from .word_tokenizer import WordTokenizer

__all__ = ["WordTokenizer", "CharTokenizer", "BigramTokenizer", "ShortWordTokenizer"]

_MAP = {
    "word": WordTokenizer,
    "char": CharTokenizer,
    "bigram": BigramTokenizer,
    "short-word": ShortWordTokenizer
}

options = list(_MAP.keys())

def getTokenizer(name: str) -> Tokenizer:
    if name.lower() in _MAP: 
        return _MAP[name.lower()]()
    else:
        raise ValueError(f"Invalid tokenizer name: {name}")
