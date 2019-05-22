import os
import gym
import pandas as pd

import tensorflow.python.framework.ops as tf1

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

from app.drl.bitcoin_trading_env import BitcoinTradingEnv

class BitcoinTradingEngine(object):
    def __init__(self):
        tf1.disable_eager_execution()
        self.name = 'BitcoinTradingEngine'
        self.model_file = './work/bitcoin_drl.a2c'
        self.train_steps = 2000
        self.test_size = 500
        self.first_run = False

    def startup(self):
        print('深度强化学习比特币量化交易平台 v0.0.1')
        model = self.train()
        self.evaluate(model)

    def build_train_env(self):
        df = pd.read_csv('./data/bitstamp.csv')
        df = df.sort_values('Timestamp')
        self.slice_point = int(len(df) - self.test_size)
        self.train_df = df[:self.slice_point]
        self.test_df = df[self.slice_point:]
        return DummyVecEnv(
            [lambda: BitcoinTradingEnv(self.train_df, serial=True)])

    def build_model(self, train_env):
        ''' build model for the very first time '''
        return A2C(MlpLstmPolicy, train_env, verbose=1,
                tensorboard_log="./tensorboard/") # ok
        '''
        return A2C(MlpLstmPolicy, train_env, verbose=1,
                tensorboard_log="./tensorboard/") # ok
        return A2C(MlpPolicy, train_env, verbose=1,
                tensorboard_log="./tensorboard/") # ok
        return A2C(CnnLnLstmPolicy, train_env, verbose=1,
                tensorboard_log="./tensorboard/")
        '''

    def train(self):
        train_env = self.build_train_env()
        if self.first_run:
            model = self.build_model(train_env)
        else:
            print('装入已有训练模型...')
            model = A2C.load(self.model_file)
            model.set_env(train_env)
        model.learn(total_timesteps=self.train_steps)
        model.save(self.model_file)
        train_env.close()
        return model

    def evaluate(self, model):
        test_env = DummyVecEnv(
                [lambda: BitcoinTradingEnv(self.test_df, serial=True)])
        obs = test_env.reset()
        print('before test')
        for i in range(self.test_size):
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            if done:
                break
            print('i={0}: {1}'.format(i, info))
            test_env.render(mode="human", title="BTC")
        print('after test')

        test_env.close()
