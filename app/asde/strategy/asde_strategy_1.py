import time
import numpy as np
import app_registry
from app_registry import appRegistry as ar
from controller.c_account import CAccount
from app.asde.ml.asde_svm import AsdeSvm
from app.asde.asde_ds import AsdeDs

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
        stock['ml_model'] = AsdeSvm()
        stock['ml_model'].train(stock['train_x'], stock['train_y'])
        print('ml_model:{0}'.format(stock['ml_model']))
        test_x = [stock['train_x'][0]]
        rst = stock['ml_model'].predict(test_x)
        print('rst:{0}'.format(rst))

    def run(self, account_id, stock, trade_date, quotation):
        '''
        根据历史行情数据，决定买卖操作，并确定购买股数
        @param stock：股票对象
        @param quotation：上一交易日行情数据
        @return direction：买卖方向；shares：股数
        @version v0.0.1 闫涛 2019-03-15
        '''
        stock['train_x'] = np.append(stock['train_x'], 
                    stock['test_x'], axis=0)
        if stock['train_x'][-1][3] / stock['train_x'][-2][3] > ar.increase_threshold:
            stock['train_y'] = np.append(stock['train_y'], 1)
        else:
            stock['train_y'] = np.append(stock['train_y'], 0)
        sample = quotation.reshape((1, 9))
        AsdeDs.normalize_datas(sample, stock['mus'], stock['stds'])
        stock['test_x'] = np.array([[
            sample[0][0], sample[0][1], sample[0][2],
            sample[0][3], sample[0][4], sample[0][5],
            sample[0][6], sample[0][7], sample[0][8]
        ]])
        stock['ml_model'].train(stock['train_x'], stock['train_y'])
        rst = stock['ml_model'].predict(stock['test_x'])
        close_price = quotation[3]
        cash_amount, stock_amount = CAccount.get_current_amounts(account_id)
        if rst[0] > 0:
            # 买入股票
            print('买入股票')
        else:
            # 卖出股票
            print('卖出股票')
        return app_registry.ASDE_BTE_BUY, 100