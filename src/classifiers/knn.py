from sklearn.neighbors import KNeighborsClassifier
from . import Classifier

class KNNClassifier(Classifier): 
    def __init__(self):
        self.model = KNeighborsClassifier()
    
    def fit(self, data, labels):
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)
    
