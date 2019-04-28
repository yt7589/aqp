import calendar
import datetime

from qstrader import settings
from qstrader.strategy.base import AbstractStrategy
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.event import SignalEvent, EventType
from qstrader.compat import queue
from qstrader.trading_session import TradingSession
from qstrader.price_handler.bscna_daily_csv_bar import BscnaDailyCsvBarPriceHandler

import matplotlib.pyplot as plt



from app.tpsa.tpsa_strategy import TpsaStrategy

class TpsaEngine(object):
    def __init__(self):
        self.name = 'QhEngine'
        
    def startup(self):
        testing = False
        config = settings.load_config()
        tickers = ['ICBC', 'CBC']
        self.run(config, testing, tickers) 

    def run(self, config, testing, tickers):
        # Backtest information
        title = [
            '基于卡尔曼滤波器的交易对策略'
        ]
        initial_equity = 500000.0
        start_date = datetime.datetime(2017, 1, 1)
        end_date = datetime.datetime(2019, 4, 23)

        # Use the Monthly Liquidate And Rebalance strategy
        events_queue = queue.Queue()
        
        
        strategy = TpsaStrategy(
            tickers, events_queue, initial_equity
        )
        
        
        etfs = ['TLT', 'IEI']
        start_date = "2010-8-01"
        end_date = "2016-08-01"
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        prices = pd.read_csv('./work/aqt005_001.txt', 
                    encoding='utf-8', parse_dates=['Date'], 
                    date_parser=dateparse, index_col='Date')
        strategy.train_kalman_filter(etfs, prices)

        # Use the Naive Position Sizer where
        # suggested quantities are followed
        position_sizer = NaivePositionSizer()


        # Set up the backtest
        backtest = TradingSession(
            config, strategy, tickers,
            initial_equity, start_date, end_date,
            events_queue, title=title,
            position_sizer=position_sizer,
            price_handler=BscnaDailyCsvBarPriceHandler(
                config.CSV_DATA_DIR, events_queue,
                tickers, start_date=start_date,
                end_date=end_date
            )
        )
        results = backtest.start_trading(testing=testing)
        print('最后金额：{0}'.format(strategy.equity))
        return results
