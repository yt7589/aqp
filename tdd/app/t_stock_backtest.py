import unittest
from app.stock_backtest import StockBacktest

class TStockBacktest(unittest.TestCase):
    def test_buy_stock(self):
        user_id = 1
        account_id = 1
        ts_code = '603912.SH'
        curr_date = '20190102'
        buy_vol = 888
        sbt = StockBacktest()
        sbt.buy_stock(user_id, account_id, ts_code, curr_date, buy_vol)
        self.assertTrue(True)

    def test_sell_stock(self):
        user_id = 1
        account_id = 1
        ts_code = '603912.SH'
        trade_date = '20190103'
        sell_vol = 800
        sbt = StockBacktest()
        sbt.sell_stock(user_id, account_id, ts_code, trade_date, sell_vol)
        self.assertTrue(True)

    def test_get_stock_vo(self):
        stock_id = 69
        ts_code = '603912.SH'
        start_dt = '20180101'
        end_dt = '20181231'
        sbt = StockBacktest()
        stock_vo = sbt.get_stock_vo(stock_id, ts_code, start_dt, end_dt)
        print('{0} {1} {2} {3} tran_len:{4} test_len:{5}'.format(
                    stock_vo['stock_id'], stock_vo['ts_code'], 
                    stock_vo['mus'], stock_vo['stds'], 
                    len(stock_vo['train_x']), len(stock_vo['test_x']))
        )
        self.assertTrue(True)
