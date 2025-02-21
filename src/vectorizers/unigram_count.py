
from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import CountVectorizer

from . import Vectorizer


class UnigramCountVectorizer(Vectorizer):
    def __init__(self, max_features:Optional[int]=None):
        super().__init__(max_features)
        self.vectorizer = CountVectorizer(analyzer="char", max_features=self.max_features, ngram_range=(1, 1))

    def fit(self, data: Iterable[List[str]]) -> None:
        self.vectorizer.fit(data)

    def transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        return self.vectorizer.transform(data)

    def fit_transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        return self.vectorizer.fit_transform(data)
