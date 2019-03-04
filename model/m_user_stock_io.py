import model.m_mysql as db

'''
股票具体买入或卖出流水记录
'''
class MUserStockIo(object):
    USER_STOCK_BUY = 1
    USER_STOCK_SELL = 2

    def __init__(self):
        self.name = 'MUserStock'

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
        return MUserStockIo._do_user_stock_io(user_stock_id, 
                    MUserStockIo.USER_STOCK_BUY, vol, price)

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
        return MUserStockIo._do_user_stock_io(user_stock_id, 
                    MUserStockIo.USER_STOCK_SELL, vol, price)

    @staticmethod
    def _do_user_stock_io(user_stock_id, io_type, vol, price):
        # 获取当前持股数量
        hold_vol = 100
        # 获取前一日收盘价格
        raw_close_price = 11.3
        close_price = raw_close_price * 100
        # 获取交易成本
        if MUserStockIo.USER_STOCK_BUY == io_type:
            buy_cost = 0.0
            buy_cost = int(buy_cost * 100)
            hold_vol += vol
            hold_amount = hold_vol * close_price - buy_cost
            params = (user_stock_id, vol, price*100, buy_cost, 
                vol * price * 100, 0, 0, 0, 0,
                hold_vol, close_price, hold_amount
            )
        else:
            sell_cost = 0.0
            sell_cost = int(sell_cost * 100)
            hold_vol -= vol
            hold_amount = hold_vol * close_price - sell_cost
            params = (user_stock_id, 0, 0, 0, 0, vol, price*100, 
                sell_cost, vol*price*100,
                hold_vol, close_price, hold_amount
            )
        sql = 'insert into t_user_stock_io(user_stock_id, buy_vol, '\
                'buy_price, buy_cost, buy_amount, sell_vol, '\
                'sell_price, sell_cost, sell_amount, hold_vol, '\
                'hold_price, hold_amount) values(%s, %s, %s, '\
                '%s, %s, %s, %s, %s, %s, %s, %s, %s)'
        return db.insert(sql, params)