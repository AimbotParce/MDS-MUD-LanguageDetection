from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray


class Vectorizer(ABC):
    "Converts lists of tokens into a single numpy array"
    def __init__(self, max_features: Optional[int] = None):
        self.max_features = max_features

    @abstractmethod
    def fit(self, data: Iterable[List[str]]) -> None:
        pass

    @abstractmethod
    def transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        pass

    @abstractmethod
    def fit_transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        pass

    @abstractmethod
    def get_vocab(self) -> List[str]:
        pass

    @abstractmethod
    def compute_coverage(self, data: Iterable[str]) -> float:
        pass


    

from .unigram_count import UnigramCountVectorizer

__all__ = ["UnigramCountVectorizer"]

_MAP = {
    "unigram-count": UnigramCountVectorizer
}

options = list(_MAP.keys())

def getVectorizer(name: str, max_features:Optional[int] = None) -> Vectorizer:
    if name.lower() in _MAP:
        return _MAP[name.lower()](max_features)
    else:
        raise ValueError(f"Invalid vectorizer name: {name}")
    

