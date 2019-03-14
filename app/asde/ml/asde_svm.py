import numpy as np
from sklearn.svm import SVC
from controller.c_stock_daily import CStockDaily

# 根据昨日开盘价、最高价、最低价、收盘价、交易量和交易金额、涨跌，
# 判断第二天股票涨跌情况
class AsdeSvm(object):
    def __init__(self):
        self.name = 'AsdeSvm'
        self.model = None
        
    def train(self, x, y):
        '''
        根据训练样本集训练SVM模型
        @param x：训练样本集输入信号
        @param y：训练样本集标签
        @version v0.0.1 闫涛 2019-03-14
        '''
        self.model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        self.model.fit(x, y)
        
    def predict(self, x):
        return self.model.predict(x)