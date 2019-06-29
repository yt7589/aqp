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

    def test_t001(self):
        self.max_min_file = './work/btc_max_min.csv'
        self.dataset_size = 10
        cached_quotation = self.t001()
        for i in range(20):
            tick = np.array([i*10+1, i*10+2, 
                    i*10+3, i*10+4, i*10+5])
            cached_quotation = self.add_quotation_tick(cached_quotation, tick)
        print(cached_quotation)

    

    def t001(self):
        return np.loadtxt(self.max_min_file, delimiter=',')

    def add_quotation_tick(self, cached_quotation, tick):
        if cached_quotation.shape[0]<self.dataset_size:
            cached_quotation = np.append(cached_quotation, [tick], axis=0)
        else:
            # 删除最前面条目
            cached_quotation = np.delete(cached_quotation, 0, axis=0)
            cached_quotation = np.append(cached_quotation, [tick], axis=0)
        return cached_quotation

    def test_t002(self):
        self.X_train = np.array([
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
            [3.1, 3.2, 3.3, 3.4, 3.5]
        ])
        x = [4.1, 4.2, 4.3, 4.4, 4.5]
        self.X_train = np.append(self.X_train, [x], axis=0)
        print(self.X_train)
        self.y_train = np.array([1, 2, 3, 4, 5])
        action = 6
        self.y_train = np.append(self.y_train, [action], axis=0)
        print(self.y_train)
        self.X_train = np.delete(self.X_train, 0, axis=0)
        print(self.X_train)