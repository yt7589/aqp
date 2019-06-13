import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from app.fme.fme_env import FmeEnv

class FmeMlpAgent(object):
    def __init__(self):
        self.name = 'FmeMlpAgent'
        self.test_size = 500

    def generate_dataset(self):
        train_env = self.build_env()
        obs = train_env.reset()
        sum = 0
        for i in range(self.slice_point):
            # 空仓样本
            action = self.choose_action(i+self.fme_env.lookback_window_size, obs, 0)
            if 0 == action[0]:
                y1 = np.array([1.0, 0.0, 0.0])
            elif 1 == action[0]:
                y1 = np.array([0.0, 1.0, 0.0])
            elif 2 == action[0]:
                y1 = np.array([0.0, 0.0, 1.0])
            # 满仓样本
            action = self.choose_action(i+self.fme_env.lookback_window_size, obs, 1)
            if 0 == action[0]:
                y2 = np.array([1.0, 0.0, 0.0])
            elif 1 == action[0]:
                y2 = np.array([0.0, 1.0, 0.0])
            elif 2 == action[0]:
                y2 = np.array([0.0, 0.0, 1.0])
            obs, rewards, done, info = train_env.step([action])
            obs_t = np.reshape(obs, (obs.shape[1], obs.shape[2])).T
            x = np.reshape(obs_t, (obs_t.shape[0]*obs_t.shape[1]))
            print('      x={0}; y1={1}; y2={2}'.format(x.shape, y1, y2))
            sum += 1
            if sum > 21:
                done = True
            if done:
                break
            train_env.render(mode="system", title="BTC")
        print('回测结束 ^_^')

    def build_env(self):
        self.df = pd.read_csv('./data/bitstamp.csv')
        self.df = self.df.dropna().reset_index()
        self.df = self.df.sort_values('Timestamp')
        self.slice_point = int(len(self.df) - self.test_size)
        self.train_df = np.array(self.df[:self.slice_point])
        self.test_df = np.array(self.df[self.slice_point:])
        lookback_window_size = 5
        self.fme_env = FmeEnv(self.df, serial=True, 
                    lookback_window_size=lookback_window_size)
        return DummyVecEnv(
            [lambda: self.fme_env])

    def choose_action(self, idx, obs, position=0):
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
                if 0 == position:
                    action = np.array([0, 10]) # 买入
                    break
                else:
                    break
            elif datas[i][close_idx] < current_price*(1-commission):
                if 1 == position:
                    action = np.array([1, 10])
                    break
                else:
                    break
        return action
        #return self.fme_env.action_space.sample()

    def test001(self):
        train_env = self.build_env()
        obs = train_env.reset()
        idx = self.fme_env.lookback_window_size + 1
        action = self.choose_action(idx, obs)
        print('action:{0}; {1}'.format(type(action), action))
        a1 = self.fme_env.action_space.sample()
        print('a1:{0}; {1}'.format(type(a1), a1))