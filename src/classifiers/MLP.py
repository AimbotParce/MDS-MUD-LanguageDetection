from sklearn.neural_network import MLPClassifier
from . import Classifier

class NNClassifier(Classifier): 
    def __init__(self):
        self.model = MLPClassifier()
    
    def fit(self, data, labels):
        
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)
        
