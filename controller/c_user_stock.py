from model.m_user_stock import MUserStock
from model.m_user_stock_io import MUserStockIo

class CUserStock(object):
    def __init__(self):
        self.name = 'CUserStock'

    @staticmethod
    def buy_user_stock(user_stock_id, vol, price):
        '''
        指定用户购买指定股票，价格为指定的价格
        @param user_stock_id：指定用户和股票编号
        @param vol：购买数量
        @param price；实际购买价格
        @return 成功或失败，失败原因
        @version v0.0.1 闫涛 2019-03-04
        '''
        pk, affected_rows = MUserStockIo.buy_user_stock(user_stock_id, vol, price)
        if affected_rows != 1:
            return False
        else:
            return True

    @staticmethod
    def sell_user_stock(user_stock_id, vol, price):
        '''
        指定用户卖出指定股票，价格为指定的价格
        @param user_stock_id：指定用户和股票编号
        @param vol：购买数量
        @param price；实际购买价格
        @return 成功或失败，失败原因
        @version v0.0.1 闫涛 2019-03-04
        '''
        pk, affected_rows = MUserStockIo.sell_user_stock(user_stock_id, vol, price)
        if affected_rows != 1:
            return False
        else:
            return True