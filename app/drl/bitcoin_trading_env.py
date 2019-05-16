import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing

class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}

    def __init__(self, df, lookback_window_size=50,
        commission=0.00075, initial_balance=10000,
        serial=False
    ):
        self.name = 'BitcoinTradingEnv'
        print('Bitcoin Trading Env is starting up...')
        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial  # Actions of the format Buy 1/10, Sell 3/10, Hold, etc.
        # Observes the OHCLV values, net worth, and trade history
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 
                    lookback_window_size + 1), dtype=np.float16)