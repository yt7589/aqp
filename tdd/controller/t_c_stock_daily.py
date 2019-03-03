import unittest
from controller.c_stock_daily import CStockDaily

class TCStockDaily(unittest.TestCase):
    def test_get_close(self):
        ts_code = '603912.SH'
        current_date = '20190103'
        close_price = CStockDaily.get_close(ts_code, current_date)
        print('{0}收盘价：{1}'.format(current_date, close_price))
        self.assertEqual('a', 'a')

if '__main__' == __name__:
    unittest.main()
