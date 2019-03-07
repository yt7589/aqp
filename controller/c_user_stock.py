from model.m_user_stock import MUserStock
from model.m_user_stock_io import MUserStockIo
from controller.c_stock import CStock
from controller.c_stock_daily import CStockDaily
from util.app_util import AppUtil

class CUserStock(object):
    def __init__(self):
        self.name = 'CUserStock'

    @staticmethod
    def buy_user_stock(user_stock_id, vol, price, buy_date):
        '''
        指定用户购买指定股票，价格为指定的价格
        @param user_stock_id：指定用户和股票编号
        @param vol：购买数量
        @param price；实际购买价格
        @return 成功或失败，失败原因
        @version v0.0.1 闫涛 2019-03-04
        '''
        pk, affected_rows = MUserStockIo.buy_user_stock(user_stock_id, vol, price, buy_date)
        if affected_rows != 1:
            return False
        else:
            return True

    @staticmethod
    def sell_user_stock(user_stock_id, vol, price, sell_date):
        '''
        指定用户卖出指定股票，价格为指定的价格
        @param user_stock_id：指定用户和股票编号
        @param vol：购买数量
        @param price；实际购买价格
        @return 成功或失败，失败原因
        @version v0.0.1 闫涛 2019-03-04
        '''
        pk, affected_rows = MUserStockIo.sell_user_stock(user_stock_id, vol, price, sell_date)
        if affected_rows != 1:
            return False
        else:
            return True

    @staticmethod
    def buy_stock_for_user(user_id, stock_id, vol, price, buy_date):
        stock_vo = CStock.get_stock_vo_by_id(stock_id)
        ts_code = stock_vo[0]
        rc, rows = MUserStock.get_user_stock_id(user_id, stock_id)
        user_stock_id = 0
        if rc <= 0:
            print('生成新记录并返回：购买日期：{0}'.format(buy_date))
            close_price = CStockDaily.get_real_close(ts_code, buy_date)
            close_price = int(close_price * 100)
            user_stock_id, _ = MUserStock.insert_user_stock(
                        user_id, stock_id, vol, close_price
            )
        else:
            user_stock_id = rows[0][0]
            hold_vol = 0
            rc, rows = MUserStock.get_user_stock_hold(user_stock_id)
            if rc > 0:
                hold_vol = rows[0][0]
            MUserStock.update_user_stock(user_id, stock_id, vol+hold_vol, close_price)
        MUserStockIo.buy_user_stock(user_stock_id, vol, price, buy_date)

    @staticmethod
    def get_user_stock_vol(user_id, stock_id):
        '''
        获取用户当前的持股量
        @param user_id；用户编号
        @param stock_id：股票编号
        @return 用户持有股票数
        @version v0.0.1 闫涛 2019-03-07
        '''
        rc, rows = MUserStock.get_user_stock_id(user_id, stock_id)
        user_stock_id = 0
        if rc > 0:
            user_stock_id = rows[0][0]
        rc, rows = MUserStock.get_user_stock_hold(user_stock_id)
        if rc > 0:
            return rows[0][0]
        else:
            return 0

    @staticmethod
    def sell_stock_for_user(user_id, stock_id, sell_vol, price, trade_date):
        '''
        卖出股票
        @param user_id：用户编号
        @param stock_id：股票编号
        @param sell_vol：卖出数量
        @param price：卖出价格
        @param trade_date：交易日期
        '''
        stock_vo = CStock.get_stock_vo_by_id(stock_id)
        ts_code = stock_vo[0]
        rc, rows = MUserStock.get_user_stock_id(user_id, stock_id)
        if rc <= 0:
            return
        user_stock_id = rows[0][0]
        hold_vol = 0
        rc, rows = MUserStock.get_user_stock_hold(user_stock_id)
        if rc > 0:
            hold_vol = rows[0][0]
        MUserStock.update_user_stock(user_id, stock_id, hold_vol-sell_vol, price)
        MUserStockIo.sell_user_stock(user_stock_id, sell_vol, price, trade_date)
        
