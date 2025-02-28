
from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import CountVectorizer

from . import Vectorizer


class TokenCountVectorizer(Vectorizer):
    """
    Counts the number of occurrences of each token in the vocabulary.
    This is a generalization of the sklearn CountVectorizer.
    """

    def __init__(self, max_features: Optional[int] = None):
        super().__init__(max_features)
        self._vectorizer = CountVectorizer(analyzer=lambda x:x, max_features=max_features)
        # If analyzer is a function, CountVectorizer will actually not split the data into tokens itself, 
        # buf it will use said function to split them. Given that we decided to perform these two steps separately,
        # we will use a lambda function that returns the input as is, so that the data is not split further than
        # it already is.

    def fit(self, data: Iterable[List[str]]) -> None:
        self._vectorizer.fit(data)

    def transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        return np.asarray(self._vectorizer.transform(data).todense(), dtype=np.float32)

    def fit_transform(self, data: Iterable[List[str]]) -> List[NDArray[np.float32]]:
        return np.asarray(self._vectorizer.fit_transform(data).todense(), dtype=np.float32)

    def get_vocab(self) -> List[str]:
        return self._vectorizer.get_feature_names_out()
