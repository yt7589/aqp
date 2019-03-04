import unittest
from app.ashare.ashare_strategy1 import AshareStrategy1

class TAshareStrategy1(unittest.TestCase):
    def test_calculate_buy_money(self):
        cash_amount = 1100000
        percent = 0.1
        price = 11.3
        result = 97
        buy_shares = AshareStrategy1.calculate_buy_money(cash_amount, percent, price)
        print('v1 购买股数：{0}；正确值：{1}'.format(buy_shares, result))
        self.assertEqual(buy_shares, result)