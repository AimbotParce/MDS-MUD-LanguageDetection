from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from . import Classifier

class LDAClassifier(Classifier):
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
    
    def fit(self, data, labels):
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)
    