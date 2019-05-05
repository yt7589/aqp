import calendar
import datetime
import numpy as np
import pickle
from qstrader import settings
from qstrader.strategy.base import AbstractStrategy
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.event import SignalEvent, EventType
from qstrader.compat import queue
from qstrader.trading_session import TradingSession
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.price_parser import PriceParser
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.price_handler.bscna_daily_csv_bar import BscnaDailyCsvBarPriceHandler

import matplotlib.pyplot as plt
from app.tpsa.tpsa_dataset import TpsaDataset
from app.tpsa.kalman_filter_engine import KalmanFilterEngine
from app.tpsa.kalman_filter_strategy import KalmanFilterStrategy

from app.tpsa.tpsa_strategy import TpsaStrategy
from app.tpsa.tpsa_risk_manager import TpsaRiskManager
from app.tpsa.kalman_filter_risk_manager import KalmanFilterRiskManager

class TpsaEngine(object):
    def __init__(self):
        self.name = 'QhEngine'
        
    def startup(self):
        testing = False
        config = settings.load_config()
        tickers = ['ICBC', 'CBC']
        engines = [KalmanFilterEngine()]
        self.run(config, testing, tickers) 

    def run(self, config, testing, tickers):
        self.title = [
            '基于卡尔曼滤波器的交易对策略'
        ]
        self.initial_equity = 1000000.0 + 4.43 * 50000 + 5.49 * 50000
        self.start_date = datetime.datetime(2017, 1, 1)
        self.end_date = datetime.datetime(2019, 4, 23)
        
        
        
        self.events_queue = queue.Queue()
        
        # 读取用于估计卡尔曼滤波参数的时间序列
        ts0 = np.array(TpsaDataset.read_close_prices('./data/{0}_train.csv'.format(tickers[0])))
        ts1 = np.array(TpsaDataset.read_close_prices('./data/{0}_train.csv'.format(tickers[1])))
        self.strategies = [KalmanFilterStrategy(tickers, self.events_queue, self.initial_equity, ts0, ts1)]
        
        self.strategy = TpsaStrategy(
            tickers, self.events_queue, self.initial_equity, self.strategies
        )
        
        # Use the Naive Position Sizer where
        # suggested quantities are followed
        position_sizer = NaivePositionSizer()
        pickle_path = './work/hmm.pkl'
        hmm_model = pickle.load(open(pickle_path, "rb"))
        risk_managers = {
            'KalmanFilterStrategy': KalmanFilterRiskManager(hmm_model)
        }
        risk_manager = TpsaRiskManager(risk_managers=risk_managers)
        '''
        price_handler=BscnaDailyCsvBarPriceHandler(
                config.CSV_DATA_DIR, self.events_queue,
                tickers, start_date=self.start_date,
                end_date=self.end_date
            )
        '''
        price_handler = BscnaDailyCsvBarPriceHandler(
            config.CSV_DATA_DIR, self.events_queue, tickers,
            start_date=self.start_date, 
            end_date=self.end_date,
            calc_adj_returns=True
        )
        
        
        
        portfolio_handler = PortfolioHandler(
            PriceParser.parse(self.initial_equity), 
            self.events_queue, price_handler,
            position_sizer, risk_manager
        )
        statistics = TearsheetStatistics(
            config, portfolio_handler, 
            self.title, benchmark="ICBC"
        )
        backtest = TradingSession(
            config, self.strategy, tickers,
            self.initial_equity, self.start_date, self.end_date,
            self.events_queue, title=self.title,
            price_handler=price_handler,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            statistics=statistics,
            portfolio_handler=portfolio_handler
        )
        results = backtest.start_trading(testing=testing)
        
        '''
        # Set up the backtest
        backtest = TradingSession(
            config, self.strategy, tickers,
            self.initial_equity, self.start_date, self.end_date,
            self.events_queue, title=self.title,
            position_sizer=position_sizer,
            price_handler=price_handler
        )
        results = backtest.start_trading(testing=testing)
        '''
        
        
        print('最后金额：{0}'.format(self.strategies[0].equity))
        
        v1 = self.strategies[0].equity
        v2 = self.strategies[0].qty0 * self.strategies[0].latest_prices[0]
        v3 = self.strategies[0].qty1 * self.strategies[0].latest_prices[1]
        t1 = v1 + v2 + v3
        print('##### test {0}={1}+{2}+{3}'.format(t1, v1, v2, v3))
        total = self.strategies[0].equity + self.strategies[0].qty0 * \
                self.strategies[0].latest_prices[0] + \
                self.strategies[0].qty1 * self.strategies[0].latest_prices[1]
        print('########### 总资产：{0}={1}+{2}+{3}'.format(
                total, self.strategies[0].equity, 
                self.strategies[0].qty0 * self.strategies[0].latest_prices[0], 
                self.strategies[0].qty1 * self.strategies[0].latest_prices[1]
        ))
        delta0 = self.strategies[0].qty0 - self.strategies[0].qty0_0
        amt0 = 0.0
        if delta0 > 0:
            amt0 = self.strategies[0].latest_prices[0] * delta0
        else:
            amt0 = -self.strategies[0].latest_prices[0] * delta0
        delta1 = self.strategies[0].qty1 - self.strategies[0].qty1_0
        amt1 = 0.0
        if delta1 > 0:
            amt1 = self.strategies[0].latest_prices[1] * delta1
        else:
            amt1 = -self.strategies[0].latest_prices[1] * delta1
            
        print('initial:{0} vs final {1}'.format(self.strategies[0].equity_0, self.strategies[0].equity+amt0+amt1))
        
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
