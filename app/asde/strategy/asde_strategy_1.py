import time
from app.asde.ml.asde_svm import AsdeSvm

# A股日线策略类
class AsdeStrategy1(object):
    def __init__(self):
        self.name = 'AsdeStrategy1'

    def setup_stock_ml_model(self, stock):
        stock['svm'] = AsdeSvm()
        stock['svm'].train(stock['train_x'], stock['train_y'])
        print('svm:{0}'.format(stock['svm']))
        test_x = [stock['train_x'][0]]
        rst = stock['svm'].predict(test_x)
        print('rst:{0}'.format(rst))