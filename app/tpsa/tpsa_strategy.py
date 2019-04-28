# kalman_qstrader_strategy.py

from __future__ import print_function

from math import floor

import numpy as np

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy


class TpsaStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    """
    def __init__(
        self, tickers, events_queue, equity
    ):
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
        self.qty = 2000
        self.cur_hedge_qty = self.qty
        
        self.yt_state = 0
        self.buy_price = 0.0

    def _set_correct_time_and_price(self, event):
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

    def train_kalman_filter(self, etfs, prices):
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
            [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
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
        
        yt = prices[etfs[1]].values
        #state_means, state_covs = kf.em(observations).filter(observations)
        xt_means, xt_covs = kf.em(yt).filter(yt)
        return xt_means, xt_covs

    def calculate_signals(self, event):
        if event.type == EventType.BAR:
            self._set_correct_time_and_price(event)
            # Only trade if we have both observations
            if all(self.latest_prices > -1.0):
                # kalman.filter_update
                pass
    
    def calculate_signals_god(self, event):
        if event.type == EventType.BAR:
            self._set_correct_time_and_price(event)

            # Only trade if we have both observations
            if all(self.latest_prices > -1.0):
                if self.days < 10 and self.latest_prices[0] < 5.0 and 0 == self.yt_state:
                    self.qty = int(500000.0 / 5.0)
                    self.equity -= self.qty * self.latest_prices[0]
                    print('日期：{0}； 操作：买入； 价格：{1}； 数量：{2}   <=> {3}!'.format(event.time, self.latest_prices[0], self.qty, self.equity))
                    self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.qty))
                    self.yt_state = 1
                    self.buy_price = self.latest_prices[0]
                if self.days > 20 and self.latest_prices[0] > 7.5 and 1 == self.yt_state:
                    print('日期：{0}； 操作：卖出； 价格：{1}； 数量：{2}!   <=> {3}'.format(event.time, self.latest_prices[0], self.qty, self.equity))
                    self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.qty))
                    self.equity += self.qty * self.latest_prices[0]
                    self.yt_state = 0
                    
    
    def calculate_signals_kalman(self, event):
        """
        Calculate the Kalman Filter strategy.
        """
        if event.type == EventType.BAR:
            self._set_correct_time_and_price(event)

            # Only trade if we have both observations
            if all(self.latest_prices > -1.0):
                # Create the observation matrix of the latest prices
                # of TLT and the intercept value (1.0) as well as the 
                # scalar value of the latest price from IEI
                F = np.asarray([self.latest_prices[0], 1.0]).reshape((1, 2))
                y = self.latest_prices[1]
            
                # The prior value of the states \theta_t is 
                # distributed as a multivariate Gaussian with 
                # mean a_t and variance-covariance R_t
                if self.R is not None:
                    self.R = self.C + self.wt
                else:
                    self.R = np.zeros((2, 2))
              
                # Calculate the Kalman Filter update
                # ----------------------------------
                # Calculate prediction of new observation
                # as well as forecast error of that prediction
                yhat = F.dot(self.theta)
                et = y - yhat

                # Q_t is the variance of the prediction of
                # observations and hence \sqrt{Q_t} is the 
                # standard deviation of the predictions
                Qt = F.dot(self.R).dot(F.T) + self.vt
                sqrt_Qt = np.sqrt(Qt)
                
                # The posterior value of the states \theta_t is
                # distributed as a multivariate Gaussian with mean
                # m_t and variance-covariance C_t
                At = self.R.dot(F.T) / Qt
                self.theta = self.theta + At.flatten() * et
                self.C = self.R - At * F.dot(self.R)

                # Only trade if days is greater than a "burn in" period
                if self.days > 1:
                    # If we're not in the market...
                    if self.invested is None:
                        if et < -sqrt_Qt:  
                            # Long Entry
                            print("LONG: %s" % event.time)
                            self.cur_hedge_qty = int(floor(self.qty*self.theta[0]))
                            self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.cur_hedge_qty))
                            self.invested = "long"
                        elif et > sqrt_Qt:  
                            # Short Entry
                            print("SHORT: %s" % event.time)
                            self.cur_hedge_qty = int(floor(self.qty*self.theta[0]))
                            self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.cur_hedge_qty))
                            self.invested = "short"
                    # If we are in the market...
                    if self.invested is not None:
                        if self.invested == "long" and et > -sqrt_Qt:
                            print("CLOSING LONG: %s" % event.time)
                            self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.cur_hedge_qty))
                            self.invested = None
                        elif self.invested == "short" and et < sqrt_Qt:
                            print("CLOSING SHORT: %s" % event.time)
                            self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.cur_hedge_qty))
                            self.invested = None