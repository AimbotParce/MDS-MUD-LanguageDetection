from sklearn.tree import DecisionTreeClassifier
from . import Classifier

class DecTreeClassifier(Classifier):
    def __init__(self):
        self.model = DecisionTreeClassifier()
        
    def fit(self, data, labels):
        self.model.fit(data, labels)
        
    def predict(self, data):
        return self.model.predict(data)