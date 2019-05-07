# kalman_qstrader_strategy.py

from __future__ import print_function
import sys
from math import floor
import math

import numpy as np
from collections import deque

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy
from pykalman import KalmanFilter


class HrmcStrategy(AbstractStrategy):
    """
    Requires:
    tickers - The list of ticker symbols
    events_queue - A handle to the system events queue
    """
    def __init__(self, tickers, 
                events_queue, base_quantity, equity,
                short_window=5, long_window=30):
        '''
        初始化隐马可夫市场机制转换趋势跟踪策略
        @param events_queue 系统事件队列
        @param base_quantity 
        @param short_window 近期移动平均窗口大小
        @param long_window 长期移动平均窗口大小
        '''
        self.name = 'HrmcStrategy'
        self.tickers = tickers
        self.events_queue = events_queue
        self.equity = equity
        self.time = None
        self.invested = None
        self.days = 0
        self.qty = 20000
        self.base_quantity = base_quantity
        self.short_window = short_window
        self.long_window = long_window
        self.bars = 0
        self.invested = False
        self.sw_bars = deque(maxlen=self.short_window)
        self.lw_bars = deque(maxlen=self.long_window)
        self.portfolio = None
        self.last_buy_price = 0.0

    def calculate_signals(self, event):
        if event.type == EventType.BAR:
            if self.time is None:
                self.time = event.time
            # Applies SMA to first ticker
            ticker = self.tickers[0]
            if event.type == EventType.BAR and event.ticker == ticker:
                # Add latest adjusted closing price to the
                # short and long window bars
                price = event.adj_close_price/float(
                    PriceParser.PRICE_MULTIPLIER
                )
                self.lw_bars.append(price)
                if self.bars > self.long_window - self.short_window:
                    self.sw_bars.append(price)
                # Enough bars are present for trading
                if self.bars > self.long_window:
                    # Calculate the simple moving averages
                    short_sma = np.mean(self.sw_bars)
                    long_sma = np.mean(self.lw_bars)
                    # Trading signals based on moving average cross
                    if short_sma > long_sma and not self.invested:
                        bot_quantity = int((self.portfolio.cur_cash / PriceParser.PRICE_MULTIPLIER * 0.9) / price)
                        signal = SignalEvent(ticker, "BOT", bot_quantity, strategy_name=self.name)
                        self.events_queue.put(signal)
                        self.invested = True
                        print('LONG: {0}; 价格：{1}; 数量：{2}'.format(event.time, price, bot_quantity))
                        self.last_buy_price = price
                    elif short_sma < long_sma and self.invested:
                        sld_quantity = 0
                        if self.tickers[0] in self.portfolio.positions:
                            sld_quantity = self.portfolio.positions[self.tickers[0]].quantity
                            if sld_quantity < 0:
                                return 
                        signal = SignalEvent(ticker, "SLD", sld_quantity, strategy_name=self.name)
                        self.events_queue.put(signal)
                        self.invested = False
                        print('SHORT: {0}; 价格：{1}; 数量：{2}'.format(event.time, price, sld_quantity))
                self.bars += 1
            
    
    
