import numpy as np

'''
策略类
'''
class AshareStrategy1(object):
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