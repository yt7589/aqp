import unittest
from app.asde.strategy.asde_strategy_1 import AsdeStrategy1
from app.asde.asde_bte import AsdeBte

class TAsdeStrategy1(unittest.TestCase):
    def test_setup_stock_ml_model(self):
        asde_bte = AsdeBte()
        stock_vo = asde_bte.get_stock_vo(69, '603912.SH', '20180101', '20181231')
        strategy = AsdeStrategy1()
        strategy.setup_stock_ml_model(stock_vo)
        self.assertTrue(True)