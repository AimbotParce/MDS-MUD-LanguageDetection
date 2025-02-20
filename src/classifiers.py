from utils import toNumpyArray

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.LogisticRegression
from sklearn.svm import SVC 


from sklearn.tree.DecisionTreeClassifier
from sklearn.ensemble.GradientBoostingClassifier
from sklearn.ensemble.RandomForest


from sklearn.discriminant_analysis.LinearDiscriminantAnalysis
from sklearn.neural_network.MLPClassifier
from sklearn.neighbors.KNeighborsClassifier

# TODO: finish this models and fill initial hyperparameters maybe?


# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test, **kwargs):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict



def applyLogisticRegression(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
def applySVM(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
def applyDecisionTree(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
def applyRandomForest(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
def applyLDA(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

def applyMLP(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

def applyKNN(X_train, y_train, X_test, **kwargs):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
