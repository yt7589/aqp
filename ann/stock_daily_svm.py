import numpy as np
from sklearn.svm import SVC
from controller.c_stock_daily import CStockDaily

class StockDailySvm(object):
    def __init__(self):
        self.name = 'Svm'
        
    @staticmethod
    def train():
        StockDailySvm.model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        StockDailySvm.model.fit(CStockDaily.train_x, CStockDaily.train_y)
        
    @staticmethod
    def predict(x):
        return StockDailySvm.model.predict(x)