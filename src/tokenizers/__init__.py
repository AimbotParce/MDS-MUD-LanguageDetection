from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray


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

from .char_tokenizer import CharTokenizer
from .word_tokenizer import WordTokenizer

__all__ = ["WordTokenizer", "CharTokenizer"]

_MAP = {
    "word": WordTokenizer,
    "char": CharTokenizer
}

options = list(_MAP.keys())

def getTokenizer(name: str) -> Tokenizer:
    if name.lower() in _MAP: 
        return _MAP[name.lower()]()
    else:
        raise ValueError(f"Invalid tokenizer name: {name}")
