from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray


class Classifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, data: NDArray[np.float32], labels: list[str]) -> None:
        pass

    @abstractmethod
    def predict(self, data: NDArray[np.float32]) -> list[str]:
        pass


from .decision_tree import DecTreeClassifier
from .knn import KNNClassifier
from .lda import LDAClassifier
from .logistic_regression import LogisticRegressionClassifier
from .MLP import NNClassifier
from .naive_bayes import NaiveBayesClassifier
from .random_forest import RdmForestClassifier
from .svm import SVMClassifier

__all__ = [
    "DecTreeClassifier",
    "KNNClassifier",
    "LDAClassifier",
    "LogisticRegressionClassifier",
    "NNClassifier",
    "NaiveBayesClassifier",
    "RdmForestClassifier",
    "SVMClassifier",
]

_MAP = {
    "dt": DecTreeClassifier,
    "knn": KNNClassifier,
    "lda": LDAClassifier,
    "lr": LogisticRegressionClassifier,
    "mlp": NNClassifier,
    "nb": NaiveBayesClassifier,
    "rf": RdmForestClassifier,
    "svm": SVMClassifier,
}

options = list(_MAP.keys())

def getClassifier(name: str) -> Classifier:
    if name.lower() in _MAP:
        return _MAP[name.lower()]()
    else:
        raise ValueError(f"Invalid classifier name: {name}")