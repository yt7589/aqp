import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn import preprocessing
from app.fme.fme_render import FmeRender

MAX_TRADING_SESSION = 100000

class FmeEnv(gym.Env):
    def __init__(self, df, lookback_window_size=50,
        commission=0.00075, initial_balance=10000,
        serial=False
    ):
        self.name = 'FmeEnv'
        print('Finacial Market Env is starting up...')
        random.seed(100)
        self.buy_rate = 1.0 # 20%机会购买
        self.sell_rate = 1.0 # 15%机会卖
        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial  # Actions of the format Buy 1/10, Sell 3/10, Hold, etc.
        # Observes the OHCLV values, net worth, and trade history
        self.scaler = preprocessing.MinMaxScaler()
        self.viewer = None
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 
                    lookback_window_size + 1), dtype=np.float16)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self._reset_session()
        self.account_history = np.repeat([
            [self.balance],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size + 1, axis=1)
        self.trades = []
        return self._next_observation()

    def _reset_session(self):
        self.current_step = 0
        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(
                self.lookback_window_size, len(self.df) - self.steps_left)
        self.active_df = self.df[self.frame_start - self.lookback_window_size:
                                 self.frame_start + self.steps_left]

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1
        scaled_df = self.active_df.values[:end].astype(np.float64)
        scaled_df = self.scaler.fit_transform(scaled_df)
        scaled_df = pd.DataFrame(scaled_df, columns=self.df.columns)
        obs = np.array([
            scaled_df['Open'].values[self.current_step:end],
            scaled_df['High'].values[self.current_step:end],
            scaled_df['Low'].values[self.current_step:end],
            scaled_df['Close'].values[self.current_step:end],
            scaled_df['Volume_(BTC)'].values[self.current_step:end],
        ])
        scaled_history = self.scaler.fit_transform(self.account_history.astype(np.float64))
        obs = np.append(
            obs, scaled_history[:, -(self.lookback_window_size + 1):], axis=0)
        return obs


    def step(self, action):
        self.current_price = self._get_current_price() + 0.01
        prev_net_worth = self.net_worth
        self._take_action(action, self.current_price)
        self.steps_left -= 1
        self.current_step += 1
        done = self.net_worth <= 0
        if self.steps_left == 0:
            print('************************ 回测结束，净资产：'
                        '{0} *********************'.format(
                        self.net_worth))
            self.balance += self.btc_held * self.current_price
            self.btc_held = 0
            self._reset_session()
            done = True
        self.obs = self._next_observation()
        reward = self.net_worth - prev_net_worth
        return self.obs, reward, done, {}

    def _get_current_price(self):
        return self.df['Close'].values[self.frame_start + 
                    self.current_step]

    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10
        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0
        if action_type < 1:
            if random.random() < self.buy_rate:
                btc_bought = self.balance / (current_price*(1+self.commission+0.001)) * amount
                btc_bought2 = self.df['Volume_(BTC)'].values[self.frame_start + self.current_step]
                if btc_bought > btc_bought2:
                    btc_bought = btc_bought2
                cost = btc_bought * current_price * (1 + self.commission)
                self.btc_held += btc_bought
                self.balance -= cost
        elif action_type < 2:
            if random.random() < self.sell_rate:
                btc_sold = self.btc_held * amount
                sales = btc_sold * current_price * (1 - self.commission)
                self.btc_held -= btc_sold
                self.balance += sales
        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.frame_start + self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'type': "sell" if btc_sold > 0 else "buy"})
        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def render(self, mode='human', **kwargs):
        if mode == 'system':
            print('Price: ' + str(self.current_price))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step + self.frame_start]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step + self.frame_start]))
            print('Net worth: ' + str(self.net_worth))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = FmeRender(
                    self.df, kwargs.get('title', None))

            print('price:{1:-10.2f} \tnet_worth:{4:-10.2f} \tbalance:{6:-10.2f} \tbtc_held:{5:-10.2f} \tBought:{2:-10.2f} \tSold:{3:-10.2f};        steps_left:{0}'.format(
                self.steps_left, 
                self.current_price,
                self.account_history[2][self.current_step + self.frame_start],
                self.account_history[4][self.current_step + self.frame_start],
                self.net_worth,
                self.btc_held,
                self.balance
            ))
            self.viewer.render(self.frame_start + self.current_step,
                               self.net_worth,
                               self.trades,
                               window_size=self.lookback_window_size)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None