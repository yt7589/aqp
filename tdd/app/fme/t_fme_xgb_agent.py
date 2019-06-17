import unittest
from app.fme.fme_xgb_agent import FmeXgbAgent

class TFmeXgbAgent(unittest.TestCase):
    def test_train_baby_agent(self):
        fme_xgb_agent = FmeXgbAgent()
        fme_xgb_agent.train_baby_agent()