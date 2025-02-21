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


from .decision_tree import DecisionTreeClassifier
from .knn import KNNClassifier
from .lda import LDAClassifier
from .logistic_regression import LogisticRegressionClassifier
from .MLP import MLPClassifier
from .naive_bayes import NaiveBayesClassifier
from .random_forest import RandomForestClassifier
from .svm import SVMClassifier

__all__ = [
    "DecisionTreeClassifier",
    "KNNClassifier",
    "LDAClassifier",
    "LogisticRegressionClassifier",
    "MLPClassifier",
    "NaiveBayesClassifier",
    "RandomForestClassifier",
    "SVMClassifier",
]


def getClassifier(name: str) -> Classifier:
    if name == "DecisionTree":
        return DecisionTreeClassifier()
    elif name == "KNN":
        return KNNClassifier()
    elif name == "LDA":
        return LDAClassifier()
    elif name == "LogisticRegression":
        return LogisticRegressionClassifier()
    elif name == "MLP":
        return MLPClassifier()
    elif name == "NaiveBayes":
        return NaiveBayesClassifier()
    elif name == "RandomForest":
        return RandomForestClassifier()
    elif name == "SVM":
        return SVMClassifier()
    else:
        raise ValueError(f"Invalid classifier name: {name}")