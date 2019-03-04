import unittest
from model.m_user_stock_io import MUserStockIo

class TMUserStockIo(unittest.TestCase):
    def test_buy_user_stock(self):
        user_stock_id = 1
        vol = 10
        price = 11.3
        pk, affected_rows = MUserStockIo.buy_user_stock(user_stock_id, vol, price)
        print('pk={0}; ar={1}'.format(pk, affected_rows))
        self.assertTrue(affected_rows==1)

    def test_sell_user_stock(self):
        user_stock_id = 1
        vol = 10
        price = 11.3
        pk, affected_rows = MUserStockIo.sell_user_stock(user_stock_id, vol, price)
        print('pk={0}; ar={1}'.format(pk, affected_rows))
        self.assertTrue(affected_rows==1)