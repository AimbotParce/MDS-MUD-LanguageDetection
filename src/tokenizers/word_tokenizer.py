from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
from nltk.tokenize import word_tokenize
from numpy.typing import NDArray


class WordTokenizer(ABC):
    "Separates sentences into lists of tokens. Can be either a static tokenizer or a trainable one"
    def __init__(self):
        pass

    def fit(self, data: Iterable[str]) -> None:
        pass

    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return list(map(word_tokenize, data)) # TODO: Word tokenizer is language-dependent!!!!!!

    @abstractmethod
    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

