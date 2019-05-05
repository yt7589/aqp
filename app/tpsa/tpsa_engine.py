import calendar
import datetime
import numpy as np
from qstrader import settings
from qstrader.strategy.base import AbstractStrategy
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.event import SignalEvent, EventType
from qstrader.compat import queue
from qstrader.trading_session import TradingSession
from qstrader.price_handler.bscna_daily_csv_bar import BscnaDailyCsvBarPriceHandler

import matplotlib.pyplot as plt
from app.tpsa.tpsa_dataset import TpsaDataset



from app.tpsa.tpsa_strategy import TpsaStrategy

class TpsaEngine(object):
    def __init__(self):
        self.name = 'QhEngine'
        
    def startup(self):
        testing = False
        config = settings.load_config()
        tickers = ['ICBC', 'CBC']
        # 读取用于估计卡尔曼滤波参数的时间序列
        ts0 = np.array(TpsaDataset.read_close_prices('./data/{0}_train.csv'.format(tickers[0])))
        ts1 = np.array(TpsaDataset.read_close_prices('./data/{0}_train.csv'.format(tickers[1])))
        
        self.title = [
            '基于卡尔曼滤波器的交易对策略'
        ]
        self.initial_equity = 1000000.0 + 4.43 * 50000 + 5.49 * 50000
        self.start_date = datetime.datetime(2017, 1, 1)
        self.end_date = datetime.datetime(2019, 4, 23)
        # Use the Monthly Liquidate And Rebalance strategy
        self.events_queue = queue.Queue()
        self.strategy = TpsaStrategy(
            tickers, self.events_queue, self.initial_equity, ts0, ts1
        )
        self.run(config, testing, tickers) 

    def run(self, config, testing, tickers):
        # Use the Naive Position Sizer where
        # suggested quantities are followed
        position_sizer = NaivePositionSizer()
        # Set up the backtest
        backtest = TradingSession(
            config, self.strategy, tickers,
            self.initial_equity, self.start_date, self.end_date,
            self.events_queue, title=self.title,
            position_sizer=position_sizer,
            price_handler=BscnaDailyCsvBarPriceHandler(
                config.CSV_DATA_DIR, self.events_queue,
                tickers, start_date=self.start_date,
                end_date=self.end_date
            )
        )
        results = backtest.start_trading(testing=testing)
        print('最后金额：{0}'.format(self.strategy.equity))
        
        v1 = self.strategy.equity
        v2 = self.strategy.qty0 * self.strategy.latest_prices[0]
        v3 = self.strategy.qty1 * self.strategy.latest_prices[1]
        t1 = v1 + v2 + v3
        print('##### test {0}={1}+{2}+{3}'.format(t1, v1, v2, v3))
        total = self.strategy.equity + self.strategy.qty0 * \
                self.strategy.latest_prices[0] + \
                self.strategy.qty1 * self.strategy.latest_prices[1]
        print('########### 总资产：{0}={1}+{2}+{3}'.format(
                total, self.strategy.equity, 
                self.strategy.qty0 * self.strategy.latest_prices[0], 
                self.strategy.qty1 * self.strategy.latest_prices[1]
        ))
        delta0 = self.strategy.qty0 - self.strategy.qty0_0
        amt0 = 0.0
        if delta0 > 0:
            amt0 = self.strategy.latest_prices[0] * delta0
        else:
            amt0 = -self.strategy.latest_prices[0] * delta0
        delta1 = self.strategy.qty1 - self.strategy.qty1_0
        amt1 = 0.0
        if delta1 > 0:
            amt1 = self.strategy.latest_prices[1] * delta1
        else:
            amt1 = -self.strategy.latest_prices[1] * delta1
            
        print('initial:{0} vs final {1}'.format(self.strategy.equity_0, self.strategy.equity+amt0+amt1))
        
        return results
        
    
    def prepare_data(self):
        stocks = [
            {
                'stock_code': 'sh.601398',
                'start_date': '2019-03-01',
                'end_date': '2019-04-29',
                'etf_name': 'ICBC'
            }, 
            {
                'stock_code': 'sh.601939',
                'start_date': '2019-03-01',
                'end_date': '2019-04-29',
                'etf_name': 'CBC'
            }
        ]
        TpsaDataset.get_quotation_data(stocks)
        stock_files = [
            './data/{0}.csv'.format(stocks[0]['etf_name']),
            './data/{0}.csv'.format(stocks[1]['etf_name'])
        ]
        TpsaDataset.draw_close_price_curve(stock_files)
