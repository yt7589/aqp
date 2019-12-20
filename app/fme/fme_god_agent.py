import numpy as np
import pandas as pd
#from stable_baselines.common.vec_env import DummyVecEnv
from app.fme.fme_env import FmeEnv

class FmeGodAgent(object):
    def __init__(self):
        self.name = 'FmeGodAgent'
        self.test_size = 500
        self.df = None
        self.fme_env = None

    def choose_action(self, idx, obs):
        commission = self.fme_env.commission
        time_span = 15
        recs = self.df.iloc[idx:idx+time_span]
        datas = np.array(recs)
        close_idx = 5
        current_price = datas[0][close_idx]
        action = np.array([2,0])
        # 判断未来涨跌
        for i in range(1, time_span):
            if datas[i][close_idx] > current_price*(1+commission):
                # 空仓时买入；满仓时持有
                if self.fme_env.btc_held <= 0.00000001:
                    action = np.array([0, 10]) # 买入
                    break
                else:
                    break
            elif datas[i][close_idx] < current_price*(1-commission):
                if self.fme_env.btc_held > 0.00000001:
                    action = np.array([1, 10])
                    break
                else:
                    break
        return action
        #return self.fme_env.action_space.sample()
