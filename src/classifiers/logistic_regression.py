from sklearn.linear_model import LogisticRegression
from . import Classifier

class LogisticRegressionClassifier(Classifier): # TODO
    def __init__(self):
        self.model = LogisticRegression()
    
    def fit(self, data, labels):
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)
    
    
