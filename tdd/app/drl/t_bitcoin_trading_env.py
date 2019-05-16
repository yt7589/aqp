import unittest
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv
from app.drl.bitcoin_trading_env import BitcoinTradingEnv

class TBitcoinTradingEnv(unittest.TestCase):
    def test_init(self):
        df = pd.read_csv('./data/bitstamp.csv')
        df = df.sort_values('Timestamp')
        slice_point = int(len(df) - 50000)
        train_df = df[:slice_point]
        test_df = df[slice_point:]
        train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_df, serial=True)])
        self.assertTrue(True)