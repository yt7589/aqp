import unittest
from controller.c_stock import CStock

class TCStock(unittest.TestCase):
    def test_get_user_stock_hold(self):
        user_stock_id = 1
        hold = CStock.get_user_stock_hold(user_stock_id)
        print('持股量为：{0}'.format(hold))
        self.assertTrue(hold > 0)