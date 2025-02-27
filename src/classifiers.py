from abc import ABC, abstractmethod

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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

__all__ = [
    "LinearDiscriminantAnalysis",
    "RandomForestClassifier",
    "LogisticRegression",
    "MultinomialNB",
    "KNeighborsClassifier",
    "MLPClassifier",
    "SVC",
    "DecisionTreeClassifier",
]

_MAP = {
    "dt": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "lda": LinearDiscriminantAnalysis,
    "lr": LogisticRegression,
    "mlp": MLPClassifier,
    "nb": MultinomialNB,
    "rf": RandomForestClassifier,
    "svm": SVC,
}

options = list(_MAP.keys())


def getClassifier(name: str) -> Classifier:
    if name.lower() in _MAP:
        return _MAP[name.lower()]()
    else:
        raise ValueError(f"Invalid classifier name: {name}")
