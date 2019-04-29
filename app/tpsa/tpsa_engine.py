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
        
        stocks = [
            {
                'stock_code': 'sh.601398',
                'start_date': '2017-01-01',
                'end_date': '2019-04-01',
                'etf_name': 'ICBC_train'
            }, 
            {
                'stock_code': 'sh.601939',
                'start_date': '2017-01-01',
                'end_date': '2019-04-01',
                'etf_name': 'CBC_train'
            }
        ]
        TpsaDataset.get_quotation_data(stocks)
        stock_files = [
            './data/{0}.csv'.format(stocks[0]['etf_name']),
            './data/{0}.csv'.format(stocks[1]['etf_name'])
        ]
        TpsaDataset.draw_close_price_curve(stock_files)
        
        ts0 = np.array(TpsaDataset.read_close_prices('./data/{0}.csv'.format(stocks[0]['etf_name'])))
        ts1 = np.array(TpsaDataset.read_close_prices('./data/{0}.csv'.format(stocks[1]['etf_name'])))
        
        title = [
            '基于卡尔曼滤波器的交易对策略'
        ]
        initial_equity = 500000.0
        start_date = datetime.datetime(2017, 1, 1)
        end_date = datetime.datetime(2019, 4, 23)

        # Use the Monthly Liquidate And Rebalance strategy
        events_queue = queue.Queue()
        
        
        strategy = TpsaStrategy(
            tickers, events_queue, initial_equity, ts0, ts1
        )
        
        print('^_^ The End ^_^')
        i_debug = 1
        if 1 == i_debug:
            return
        
        
        
        
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
