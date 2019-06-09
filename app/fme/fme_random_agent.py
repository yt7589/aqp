import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from app.fme.fme_env import FmeEnv

class FmeRandomAgent(object):
    def __init__(self):
        self.name = 'FmeAgent'
        self.test_size = 500

    def train(self):
        train_env = self.build_train_env()
        obs = train_env.reset()
        for i in range(self.slice_point):
            action = self.choose_action(i+self.fme_env.lookback_window_size, obs)
            obs, rewards, done, info = train_env.step([action])
            if done:
                break
            train_env.render(mode="human", title="BTC")
        print('回测结束 ^_^')

    def build_train_env(self):
        self.df = pd.read_csv('./data/bitstamp.csv')
        self.df = self.df.dropna().reset_index()
        self.df = self.df.sort_values('Timestamp')
        self.slice_point = int(len(self.df) - self.test_size)
        self.train_df = self.df[:self.slice_point]
        self.test_df = self.df[self.slice_point:]
        self.fme_env = FmeEnv(self.train_df, serial=True)
        return DummyVecEnv(
            [lambda: self.fme_env])

    def choose_action(self, idx, obs):
        return self.fme_env.action_space.sample()

    def test001(self):
        train_env = self.build_train_env()
        obs = train_env.reset()
        idx = self.fme_env.lookback_window_size + 1
        action = self.choose_action(idx, obs)
        print('action:{0}; {1}'.format(type(action), action))
        a1 = self.fme_env.action_space.sample()
        print('a1:{0}; {1}'.format(type(a1), a1))