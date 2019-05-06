# kalman_qstrader_strategy.py

from __future__ import print_function

from math import floor
import math

import numpy as np

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy
from pykalman import KalmanFilter


class KalmanFilterStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    """
    def __init__(
        self, tickers, events_queue, equity, ts0, ts1
    ):
        self.name = 'KalmanFilterStrategy'
        self.tickers = tickers
        self.events_queue = events_queue
        self.equity = equity
        self.time = None
        self.latest_prices = np.array([-1.0, -1.0])
        self.invested = None

        self.delta = 1e-4
        self.wt = self.delta / (1 - self.delta) * np.eye(2)
        self.vt = 1e-3
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None

        self.days = 0
        self.qty = 20000
        self.cur_hedge_qty = self.qty
        
        self.yt_state = 0
        self.buy_price = 0.0
        
        self.ts0 = ts0
        self.ts1 = ts1
        self.kalman_mode = 1
        
        xt_means, xt_covs = self.train_kalman_filter(ts0, ts1)        
        self.qty1 = 50000
        self.qty1_0 = 50000
        self.qty0 = 50000
        self.qty0_0 = 50000
        amt1 = self.qty1 * ts1[0]
        amt0 = self.qty0 * ts0[0]
        self.equity -= (amt0 + amt1)
        self.equity_0 = self.equity
        
        
        self.ts0 = np.array([])
        self.ts1 = np.array([])
        self.deltas = np.array([])
        
        self.portfolio = None
        
    def handle_event(self, event):
        """
        Sets the correct price and event time for prices
        that arrive out of order in the events queue.
        """
        # Set the first instance of time
        if self.time is None:
            self.time = event.time
        
        # Set the correct latest prices depending upon 
        # order of arrival of market bar event
        price = event.adj_close_price/float(
            PriceParser.PRICE_MULTIPLIER
        )
        if event.time == self.time:
            if event.ticker == self.tickers[0]:
                self.latest_prices[0] = price
            else:
                self.latest_prices[1] = price
        else:
            self.time = event.time
            self.days += 1
            self.latest_prices = np.array([-1.0, -1.0])
            if event.ticker == self.tickers[0]:
                self.latest_prices[0] = price
            else:
                self.latest_prices[1] = price

    def train_kalman_filter(self, ts0, ts1, mode=1):
        """
        Utilise the Kalman Filter from the PyKalman package
        to calculate the slope and intercept of the regressed
        ETF prices.
        """
        delta = 1e-5
        mu0 = np.zeros(2)
        sigma0 = np.ones((2, 2))
        Q = delta / (1 - delta) * np.eye(2)
        At = np.eye(2)
        Ct = np.vstack(
            [ts0, np.ones(ts0.shape)]
        ).T[:, np.newaxis]
        R = 1.0
        self.kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=mu0,
            initial_state_covariance=sigma0,
            transition_matrices=At,
            observation_matrices=Ct,
            observation_covariance=R,
            transition_covariance=Q
        )
        yt = ts1
        #state_means, state_covs = kf.em(observations).filter(observations)
        if mode != 1:
            xt_means, xt_covs = self.kf.filter(yt)
        else:
            xt_means, xt_covs = self.kf.em(yt).filter(yt)
        return xt_means, xt_covs


    def calculate_signals(self, event):
        mode = 2
        # kalman.filter_update
        self.ts0 = np.append(self.ts0, self.latest_prices[0])
        self.ts1 = np.append(self.ts1, self.latest_prices[1])
        if self.days < 100:
            return
        if self.days % 30 == 0:
            mode = 1
        xt_means, x_convs = self.train_kalman_filter(self.ts0, self.ts1, mode=2)
        slope = xt_means[-1][0]
        intercept = xt_means[-1][1]
        yt_hat = self.ts0[-1] * slope + intercept
        delta = yt_hat - self.ts1[-1]
        self.deltas = np.append(self.deltas, delta)
        threshold = np.std(self.deltas)
        if delta < -threshold:
            # 卖掉0买入1
            qty0 = int(math.floor(self.qty * slope))
            if self.tickers[0] not in self.portfolio.positions:
                return
            # 检查是否还有股票可卖
            if qty0 > self.portfolio.positions[self.tickers[0]].quantity:
                return
            qty1 = self.qty
            if self.portfolio.cur_cash + qty0 * self.latest_prices[0] < 1.2 * qty1 * self.latest_prices[1]:
                return
            # 满足合法性检查后，才能执行对冲操作
            self.events_queue.put(SignalEvent(self.tickers[0], "SLD", qty0, strategy_name=self.name))
            self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty, strategy_name=self.name))
            print('\r\n    买入{0}：数量：{1}；价格：{2}；金额：{3}'.format(
                    self.tickers[1], self.qty, self.latest_prices[1], 
                    self.qty*self.latest_prices[1])
            )
            print('     卖出{0}：数量：{1}；价格：{2}；金额：{3}'.format(
                    self.tickers[0], qty0, self.latest_prices[0],
                    qty0*self.latest_prices[0]
            ))
        elif delta > threshold:
            qty0 = int(math.floor(self.qty * slope))
            if self.tickers[1] not in self.portfolio.positions:
                return
            if self.portfolio.positions[self.tickers[1]].quantity - self.qty < 0:
                return
            if self.portfolio.cur_cash + self.qty * self.latest_prices[1] < 1.2*qty0 * self.latest_prices[0]:
                return
            # 买入0卖出1
            self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty, strategy_name=self.name))
            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", qty0, strategy_name=self.name))
            print('\r\n    买入{0}：数量：{1}；价格：{2}；金额：{3}'.format(
                    self.tickers[0], qty0, self.latest_prices[0], 
                    self.qty*self.latest_prices[0])
            )
            print('     卖出{0}：数量：{1}；价格：{2}；金额：{3}'.format(
                    self.tickers[1], self.qty, self.latest_prices[1],
                    qty0*self.latest_prices[1]
            ))

