import unittest
import numpy as np
from util.app_util import AppUtil
from app.asde.strategy.asde_strategy_1 import AsdeStrategy1
from app.asde.asde_bte import AsdeBte

class TAsdeStrategy1(unittest.TestCase):
    def test_setup_stock_ml_model(self):
        asde_bte = AsdeBte()
        stock_vo = asde_bte.get_stock_vo(69, '603912.SH', '20180101', '20181231')
        strategy = AsdeStrategy1()
        strategy.setup_stock_ml_model(stock_vo)
        self.assertTrue(True)

    def test_run(self):
        user_id = 1
        account_id = 1
        start_dt = '20180101'
        end_dt = '20181231'
        asde_bte = AsdeBte()
        stock = asde_bte.get_stock_vo(69, '603912.SH', start_dt, end_dt)
        strategy = AsdeStrategy1()
        strategy.setup_stock_ml_model(stock)
        trade_date = AppUtil.parse_date('20190102')
        quotation = np.array([1.3320000e+01, 1.3950000e+01, 1.3320000e+01, 1.3700000e+01, 1.3360000e+01,
 3.4000000e-01, 2.5449000e+00, 4.0978000e+04, 5.5958183e+04])
        direction, vol = strategy.run(user_id, account_id, stock, trade_date, quotation)
        print('{0}    {1}'.format(direction, vol))
        self.assertTrue(True)

    def test_calculate_buy_vol(self):
        cash_amount = 100000
        price = 1280
        strategy = AsdeStrategy1()
        buy_vol = strategy.calculate_buy_vol(cash_amount, price)
        self.assertEqual(buy_vol, 7)

    def test_calculate_sell_vol(self):
        stock_vol = 2832
        strategy = AsdeStrategy1()
        sell_vol = strategy.calculate_sell_vol(stock_vol)
        self.assertEqual(sell_vol, 283)
