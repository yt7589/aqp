import unittest
from controller.c_user_stock import CUserStock

class TCUserStock(unittest.TestCase):
    def test_buy_user_stock(self):
        user_stock_id = 1
        vol = 10
        price = 13.88
        rst = CUserStock.buy_user_stock(user_stock_id, vol, price)
        if rst:
            self.assertTrue(True)
        else:
            self.assertFalse(False)

    def test_sell_user_stock(self):
        user_stock_id = 1
        vol = 10
        price = 13.88
        rst = CUserStock.sell_user_stock(user_stock_id, vol, price)
        if rst:
            self.assertTrue(True)
        else:
            self.assertFalse(False)