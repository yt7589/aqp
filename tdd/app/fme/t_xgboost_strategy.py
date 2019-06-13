import unittest
from app.fme.xgboost_strategy import XgboostStrategy

class TXgboostStrategy(unittest.TestCase):
    def test_startup(self):
        xgboost_strategy = XgboostStrategy()
        xgboost_strategy.demo()