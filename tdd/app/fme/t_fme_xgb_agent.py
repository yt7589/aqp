import unittest
import numpy as np
import xgboost as xgb
from app.fme.fme_dataset import FmeDataset
from app.fme.fme_xgb_agent import FmeXgbAgent

class TFmeXgbAgent(unittest.TestCase):
    def test_init(self):
        fme_xgb_agent = FmeXgbAgent()

    def test_train_baby_agent(self):
        #fme_dataset = FmeDataset()
        #fme_dataset.create_bitcoin_dataset(dataset_size=10000)
        fme_xgb_agent = FmeXgbAgent()
        fme_xgb_agent.train_baby_agent()

    def test_bug(self):
        fme_xgb_agent = FmeXgbAgent()
        fme_dataset = FmeDataset()
        X, y = fme_dataset.load_bitcoin_dataset()
        self.predict(fme_xgb_agent, X, 0)
        self.predict(fme_xgb_agent, X, 4)
        self.predict(fme_xgb_agent, X, 8)
        self.predict(fme_xgb_agent, X, 9)
        self.predict(fme_xgb_agent, X, 10)

    def predict(self, fme_xgb_agent, X, idx):
        x1 = np.array([X[idx]])
        xg1 = xgb.DMatrix( x1, label=x1)
        pred = fme_xgb_agent.model.predict( xg1 )
        print('预测结果{0}：{1}'.format(idx, np.argmax(pred)))