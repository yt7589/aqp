import numpy as np
from controller.c_stock_daily import CStockDaily

'''
策略类
'''
class AshareStrategy1(object):
    models = []
    
    def __init__(self):
        self.name = 'AshareStrategy1'

    @staticmethod
    def calculate_buy_money(cash_amount, percent, price):
        '''
        根据现金、比例和股价决定购买的数量
        @param cash_amount：现金资产数量
        @param percent：现金中多少比例用于本次交易
        @param price：当前股票交易价格
        @return 返回要购买的数量
        @version v0.0.1 闫涛 2019-03-03
        '''
        money = cash_amount * percent * 0.01 # 10%用于交易，单位变为元
        shares = money / price
        buy_shares = int(shares)
        buy_money = int(buy_shares * price * 100)
        return buy_shares

    @staticmethod
    def initialize():
        '''
        初始化预测模型
        @version v0.0.1 闫涛 2019-03-08
        '''
        pass

    @staticmethod
    def run():
        '''
        以每天的行情数据作为输入，根据量化交易模型，决定指定股票的买入或卖出，以及
        买入或卖出的数量
        @version v0.0.1 闫涛 2019-03-08
        '''
        pass