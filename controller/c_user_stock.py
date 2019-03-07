from model.m_user_stock import MUserStock
from model.m_user_stock_io import MUserStockIo
from controller.c_stock import CStock
from controller.c_stock_daily import CStockDaily
from util.app_util import AppUtil

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

    @staticmethod
    def buy_stock_for_user(user_id, stock_id, vol, price):
        stock_vo = CStock.get_stock_vo_by_id(stock_id)
        ts_code = stock_vo[0]
        curr_date = AppUtil.get_current_date_str()
        rc, rows = MUserStock.get_user_stock_id(user_id, stock_id)
        if rc <= 0:
            print('生成新记录并返回')
            close_price = CStockDaily.get_real_close(ts_code, curr_date)
            MUserStock.insert_user_stock(user_id, stock_id, vol, close_price)
        else:
            user_stock_id = rows[0][0]
            hold_vol = MUserStock.get_user_stock_hold(user_stock_id)
            MUserStock.update_user_stock(user_id, stock_id, vol+hold_vol, close_price)
        MUserStockIo.buy_user_stock(user_stock_id, vol, price)

    @staticmethod
    def get_user_stock_vol(user_id, stock_id):
        '''
        获取用户当前的持股量
        '''
        rc, rows = MUserStock.get_user_stock_id(user_id, stock_id)
        user_stock_id = 0
        if rc > 0:
            user_stock_id = rows[0][0]
        return MUserStock.get_user_stock_hold(user_stock_id)
        
