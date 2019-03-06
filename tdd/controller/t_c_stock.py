import unittest
from controller.c_stock import CStock

class TCStock(unittest.TestCase):
    def test_get_user_stock_hold(self):
        user_stock_id = 1
        hold = CStock.get_user_stock_hold(user_stock_id)
        print('持股量为：{0}'.format(hold))
        self.assertTrue(hold > 0)

    def test_get_prev_day_close_price(self):
        ts_code = '603912.SH'
        curr_date = '20190103'
        close_price = CStock.get_prev_day_close_price(ts_code, curr_date)
        print('前日收盘价：{0}'.format(close_price))
        self.assertTrue(True)

    def test_get_prev_day_close_price1(self):
        ts_code = '603912.SH'
        curr_date = '20190102'
        close_price = CStock.get_prev_day_close_price(ts_code, curr_date)
        print('前日收盘价：{0}'.format(close_price))
        self.assertTrue(True)

    def test_get_stock_vo_of_user(self):
        user_stock_id = 1
        vo = CStock.get_stock_vo_of_user(user_stock_id)
        if len(vo) <= 0:
            self.assertFalse(False)
        else:
            print('股票编号：{0}；股票编码：{1}；股票代码：{2}；'\
                        '股票名称：{3}'.format(
                            vo[0], vo[1], vo[2], vo[3]
                        ))

    def test_get_stock_vo_by_id(self):
        stock_id = 69
        vo = CStock.get_stock_vo_by_id(stock_id)
        if len(vo)<=0:
            self.assertFalse(False)
        else:
            print('{0} - {1} - {2}'.format(vo[0], vo[1], vo[2]))
            self.assertTrue(True)