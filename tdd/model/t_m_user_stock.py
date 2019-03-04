import unittest
from model.m_user_stock import MUserStock

class TMUserStock(unittest.TestCase):
    def test_get_user_stock_id(self):
        user_id = 1
        stock_id = 69
        rc, rows = MUserStock.get_user_stock_id(user_id, stock_id)
        if rc <= 0:
            self.assertTrue(False)
            return
        user_stock_id = rows[0][0]
        print('user_stock_id={0}'.format(user_stock_id))
        self.assertTrue(True)

    def test_get_user_stock_hold(self):
        user_stock_id = 1
        rc, rows = MUserStock.get_user_stock_hold(user_stock_id)
        if rc <= 0:
            self.assertTrue(False)
            return
        hold = rows[0][0]
        print('持股数量：{0}'.format(hold))
        self.assertTrue(True)