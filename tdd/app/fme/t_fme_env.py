import unittest
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from app.fme.fme_env import FmeEnv

class TFmeEnv(unittest.TestCase):
    def test_init001(self):
        ''' 测试action_space和observation_space '''
        fme_evn = FmeEnv(60)
        print('action_space:{0}; {1}'.format(type(fme_evn.action_space), 
                    fme_evn.action_space))
        a1 = fme_evn.action_space.sample()
        print('a1:{0}; {1}'.format(type(a1), a1))
        print('observation_space:{0}; {1}'.format(
            type(fme_evn.observation_space), fme_evn.observation_space)
        )

    def test_init002(self):
        ''' 测试完整的构造函数 '''
        test_size = 500
        df = pd.read_csv('./data/bitstamp.csv')
        df = df.sort_values('Timestamp')
        self.slice_point = int(len(df) - test_size)
        self.train_df = df[:self.slice_point]
        self.test_df = df[self.slice_point:]
        env = DummyVecEnv(
            [lambda: FmeEnv(self.train_df, serial=True)])
        self.assertTrue(1>0)

    def test_reset001(self):
        test_size = 500
        df = pd.read_csv('./data/bitstamp.csv')
        df = df.sort_values('Timestamp')
        self.slice_point = int(len(df) - test_size)
        self.train_df = df[:self.slice_point]
        self.test_df = df[self.slice_point:]
        env = DummyVecEnv(
            [lambda: FmeEnv(self.train_df, serial=True)])
        self.assertTrue(1>0)
        env.reset()

    def test_step001(self):
        test_size = 500
        df = pd.read_csv('./data/bitstamp.csv')
        df = df.sort_values('Timestamp')
        self.slice_point = int(len(df) - test_size)
        self.train_df = df[:self.slice_point]
        self.test_df = df[self.slice_point:]
        fme_env = FmeEnv(self.train_df, serial=True)
        env = DummyVecEnv(
            [lambda: fme_env])
        env.reset()
        print('Before net_worth:{0}'.format(fme_env.net_worth))
        a1 = fme_env.action_space.sample()
        print('a1:{0}; {1}'.format(type(a1), a1))
        fme_env.step(a1)
        print('After net_worth:{0}'.format(fme_env.net_worth))



