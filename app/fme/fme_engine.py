# 控制环境运行，通知agent环境状态，执行agent所选择的操作
import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from app.fme.fme_env import FmeEnv
from app.fme.fme_god_agent import FmeGodAgent

class FmeEngine(object):
    def __init__(self):
        self.name = 'FmeEngine'
        self.env = None
        self.agent = FmeGodAgent()
        self.test_size = 1000
    
    def startup(self):
        self.env = self.build_raw_env()
        obs = self.env.reset()
        for i in range(self.slice_point):
            action = self.agent.choose_action(i+self.fme_env.lookback_window_size, obs)
            obs, rewards, done, info = self.env.step([action])
            if done:
                break
            self.env.render(mode="human", title="BTC")
        print('回测结束 ^_^')

    def build_env(self):
        ''' 创建基于数据集的深度强化学习环境  '''
        pass

    def build_raw_env(self):
        ''' 创建原始比特币行情文件生成的env，主要用于深度强化学习试验 '''
        self.df = pd.read_csv('./data/bitstamp.csv')
        self.df = self.df.dropna().reset_index()
        self.df = self.df.sort_values('Timestamp')
        self.agent.df = self.df
        self.slice_point = int(len(self.df) - self.test_size)
        self.train_df = self.df[:self.slice_point]
        self.test_df = self.df[self.slice_point:]
        self.fme_env = FmeEnv(self.train_df, serial=True)
        self.agent.fme_env = self.fme_env
        return DummyVecEnv(
            [lambda: self.fme_env])