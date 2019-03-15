import time
from app.asde.ml.asde_svm import AsdeSvm

# A股日线策略类
class AsdeStrategy1(object):
    def __init__(self):
        self.name = 'AsdeStrategy1'

    def setup_stock_ml_model(self, stock):
        '''
        初始化每支股票的机器学习模型，在本策略中使用支撑向量机
        @param stock：股票值对象，有样本集
        @version v0.0.1 闫涛 2019-03-15
        '''
        stock['svm'] = AsdeSvm()
        stock['svm'].train(stock['train_x'], stock['train_y'])
        print('svm:{0}'.format(stock['svm']))
        test_x = [stock['train_x'][0]]
        rst = stock['svm'].predict(test_x)
        print('rst:{0}'.format(rst))