import unittest
from controller.c_user_stock import CUserStock
from util.app_util import AppUtil

class TCUserStock(unittest.TestCase):
    def test_buy_user_stock(self):
        user_stock_id = 1
        vol = 10
        price = 13.88
        buy_date = AppUtil.parse_date('20190101')
        rst = CUserStock.buy_user_stock(user_stock_id, vol, price, buy_date)
        if rst:
            self.assertTrue(True)
        else:
            self.assertFalse(False)

    def test_sell_user_stock(self):
        user_stock_id = 1
        vol = 10
        price = 13.88
        sell_date = AppUtil.parse_date('20190101')
        rst = CUserStock.sell_user_stock(user_stock_id, vol, price, sell_date)
        if rst:
            self.assertTrue(True)
        else:
            self.assertFalse(False)

    def test_get_user_stock_vol(self):
        user_id = 1
        stock_id = 69
        hold_vol = CUserStock.get_user_stock_vol(user_id, stock_id)
        print('持股量：{0}'.format(hold_vol))
        self.assertTrue(True)

    def test_get_user_stocks(self):
        user_id = 1
        stocks = CUserStock.get_user_stocks(user_id)
        print(stocks)
        self.assertTrue(True)