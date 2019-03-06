import unittest
from model.m_stock import MStock

class TMStock(unittest.TestCase):
    def test_get_stock_vo_by_id(self):
        stock_id = 69
        rc, rows = MStock.get_stock_vo_by_id(stock_id)
        if rc <= 0:
            self.assertFalse(False)
        else:
            print('{0} - {1} - {2}'.format(rows[0][0], 
                    rows[0][1], rows[0][2]))
            self.assertTrue(True)

    def test_get_stock_id_by_ts_code(self):
        ts_code = '603912.SH'
        rc, rows = MStock.get_stock_id_by_ts_code(ts_code)
        if rc <= 0:
            self.assertFalse(False)
        else:
            print('股票编号：{0}'.format(rows[0][0]))
            self.assertTrue(True)