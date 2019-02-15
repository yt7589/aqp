'''
from sklearn import svm

class Svm(object):
    def __init__(self):
        self.name = 'Svm'
        
    @staticmethod
    def train(train_x, train_y, validate_x, validate_y):
        model = svm.SVC()
        model.fit(train_x, train_y)
        return model
        
    @staticmethod
    def predict(model, test_x):
        return model.predict(test_x)
'''
import numpy as np
from sklearn.svm import SVC
from controller.c_stock_daily import CStockDaily

class Svm(object):
    def __init__(self):
        self.name = 'Svm'
        
    @staticmethod
    def train():
        Svm.model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        Svm.model.fit(CStockDaily.train_x, CStockDaily.train_y)
        
    @staticmethod
    def predict(x):
        return Svm.model.predict(x)