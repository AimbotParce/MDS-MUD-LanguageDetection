from sklearn.naive_bayes import MultinomialNB

from . import Classifier


class NaiveBayesClassifier(Classifier):
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)
    
