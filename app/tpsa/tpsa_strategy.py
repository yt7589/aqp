# kalman_qstrader_strategy.py

from __future__ import print_function

from math import floor
import math

import numpy as np

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy
from pykalman import KalmanFilter


class TpsaStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    """
    def __init__(
        self, tickers, events_queue, equity, strategies=None
    ):
        self.name = 'TpsaStrategy'
        self.tickers = tickers
        self.events_queue = events_queue
        self.equity = equity
        self.time = None
        self.latest_prices = np.array([-1.0, -1.0])
        self.invested = None
        self.strategies = strategies
        self.days = 0
        self.qty = 20000
        self.cur_hedge_qty = self.qty
        
        

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


    def calculate_signals(self, event):
        mode = 2
        if event.type == EventType.BAR:
            self._set_correct_time_and_price(event)
            for strategy in self.strategies:
                strategy._set_correct_time_and_price( event)
            # Only trade if we have both observations
            if all(self.latest_prices > -1.0):
                for strategy in self.strategies:
                    strategy.calculate_signals( event)
    
    
