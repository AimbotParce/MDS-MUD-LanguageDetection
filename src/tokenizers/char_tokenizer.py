from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray


class CharTokenizer(ABC):
    "Separates sentences into lists of tokens. Can be either a static tokenizer or a trainable one"
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: Iterable[str]) -> None:
        pass

    @abstractmethod
    def transform(self, data: Iterable[str]) -> List[List[str]]:
        return list(map(list, data))

    @abstractmethod
    def fit_transform(self, data: Iterable[str]) -> List[List[str]]:
        return self.transform(data)

