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
