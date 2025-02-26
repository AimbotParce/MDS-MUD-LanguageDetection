from sklearn.ensemble import RandomForestClassifier
from . import Classifier

class RdmForestClassifier(Classifier): # TODO
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def fit(self, data, labels):
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)
    
    
