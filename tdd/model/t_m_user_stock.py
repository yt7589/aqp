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

    def test_get_stock_vo(self):
        user_stock_id = 1
        rc, rows = MUserStock.get_stock_vo(user_stock_id)
        if rc <= 0:
            self.assertFalse(False)
        else:
            print('股票编号:{0}； 股票编码：{1}; 股票代码：{2}；'\
                        '股票名称：{3}'.format(rows[0][0],
                            rows[0][1],
                            rows[0][2],
                            rows[0][3]
                        ))
            self.assertTrue(True)

    def test_insert_user_stock(self):
        user_id = 1
        stock_id = 69
        vol = 88
        price = 1100
        pk, affected_rows = MUserStock.insert_user_stock(user_id, stock_id, vol, price)
        print('主键：{0}; ar={1}'.format(pk, affected_rows))
        if pk <= 0:
            self.assertFalse(False)
        else:
            self.assertTrue(True)

    def test_update_user_stock(self):
        user_id = 1
        stock_id = 69
        vol = 18
        price = 2828
        pk, affected_rows = MUserStock.update_user_stock(user_id, stock_id, vol, price)
        if affected_rows==1 :
            self.assertTrue(True)
        else:
            self.assertFalse(False)

    def test_get_user_stocks(self):
        user_id = 1
        rc, rows = MUserStock.get_user_stocks(user_id)
        print(rows)
        self.assertTrue(True)