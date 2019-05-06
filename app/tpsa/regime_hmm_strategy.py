# regime_hmm_strategy.py

from __future__ import print_function

from collections import deque

import numpy as np

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy

class RegimeHmmStrategy(AbstractStrategy):
    def __init__(self, tickers, 
                events_queue, base_quantity,
                short_window=1, long_window=5):
        '''
        初始化隐马可夫市场机制转换趋势跟踪策略
        @param events_queue 系统事件队列
        @param base_quantity 
        @param short_window 近期移动平均窗口大小
        @param long_window 长期移动平均窗口大小
        '''
        self.name = 'RegimeHmmStrategy'
        self.tickers = tickers
        self.events_queue = events_queue
        self.base_quantity = base_quantity
        self.short_window = short_window
        self.long_window = long_window
        self.bars = 0
        self.invested = False
        self.sw_bars = deque(maxlen=self.short_window)
        self.lw_bars = deque(maxlen=self.long_window)
        self.time = None
        self.days = 0
        self.latest_prices = np.array([-1.0, -1.0])
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

    def calculate_signals(self, event):
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
                    bot_quantity = int((self.portfolio.cur_cash / PriceParser.PRICE_MULTIPLIER * 0.9) /self.latest_prices[0])
                    signal = SignalEvent(ticker, "BOT", bot_quantity, strategy_name=self.name)
                    self.events_queue.put(signal)
                    self.invested = True
                    print('LONG: {0}; 价格：{1}; 数量：{2}'.format(event.time, self.latest_prices[0], bot_quantity))
                elif short_sma < long_sma and self.invested:
                    sld_quantity = 0
                    if self.tickers[0] in self.portfolio.positions:
                        sld_quantity = self.portfolio.positions[self.tickers[0]].quantity
                        if sld_quantity < 0:
                            return 
                    signal = SignalEvent(ticker, "SLD", sld_quantity, strategy_name=self.name)
                    self.events_queue.put(signal)
                    self.invested = False
                    print('SHORT: {0}; 价格：{1}; 数量：{2}'.format(event.time, self.latest_prices[0], sld_quantity))
            self.bars += 1
        
    