import unittest
from app.fme.fme_dataset import FmeDataset
from app.fme.fme_xgb_agent import FmeXgbAgent

class TFmeXgbAgent(unittest.TestCase):
    def test_train_baby_agent(self):
        #fme_dataset = FmeDataset()
        #fme_dataset.create_bitcoin_dataset(dataset_size=10000)
        fme_xgb_agent = FmeXgbAgent()
        fme_xgb_agent.train_baby_agent()