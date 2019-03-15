import unittest
from controller.c_stock_daily import CStockDaily
from util.app_util import AppUtil

class TCStockDaily(unittest.TestCase):
    def test_get_close(self):
        ts_code = '603912.SH'
        current_date = '20190103'
        close_price = CStockDaily.get_close(ts_code, current_date)
        print('{0}收盘价：{1}'.format(current_date, close_price))
        self.assertEqual('a', 'a')

    def test_get_real_close(self):
        ts_code = '603912.SH'
        close_price = CStockDaily.get_real_close(ts_code, AppUtil.get_current_date_str())
        print('收盘价：{0}'.format(close_price))
        self.assertTrue(True)

    def test_get_daily_quotation(self):
        ts_code = '603912.SH'
        ask_date = AppUtil.parse_date('20181230')
        trade_date, quotation = CStockDaily.get_daily_quotation(ts_code, ask_date)
        print('trade_date:{0}'.format(trade_date))
        print('quotation:{0}'.format(quotation))
        self.assertTrue(True)


if '__main__' == __name__:
    unittest.main()
