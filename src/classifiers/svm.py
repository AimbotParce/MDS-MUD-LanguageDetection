from sklearn.svm import SVC
from . import Classifier

class SVMClassifier(Classifier): 
    def __init__(self):
        self.model = SVC()
    
    def fit(self, data, labels):
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)
    
    
