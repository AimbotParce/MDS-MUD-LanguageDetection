from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray

from . import Vectorizer


class TokenCountVectorizer(Vectorizer):
    "Counts the number of occurrences of each token in the vocabulary"

    def __init__(self, max_features: Optional[int] = None):
        super().__init__(max_features)
        self._token_counts: Dict[str, int] = defaultdict(int)
        self._all_tokens: set[str] = {}
        self._vocab:list[str] = None
        self._vocab_index: Dict[str, int] = None

    def fit(self, data: Iterable[List[str]]) -> None:
        for sentence in data:
            for token in sentence:
                self._token_counts[token] = 1
                self._all_tokens.add(token)
        self._vocab = sorted(self._all_tokens, key=lambda x: self._token_counts[x], reverse=True)[: self.max_features]
        self._vocab_index = {token: i for i, token in enumerate(self._vocab)}

    def transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        if self._vocab is None:
            raise ValueError("Vectorizer not fitted")
        return [np.array([sentence.count(token) for token in self._vocab], dtype=np.float32) for sentence in data]

    def fit_transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        self.fit(data)
        return self.transform(data)

    def get_vocab(self) -> List[str]:
        if self._vocab is None:
            raise ValueError("Vectorizer not fitted")
        return self._vocab
