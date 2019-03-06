import unittest
from app.stock_backtest import StockBacktest

class TStockBacktest(unittest.TestCase):
    def test_buy_stock(self):
        user_id = 1
        account_id = 1
        ts_code = '603912.SH'
        curr_date = '20190102'
        sbt = StockBacktest()
        sbt.buy_stock(user_id, account_id, ts_code, curr_date)
        self.assertTrue(True)