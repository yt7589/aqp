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

from app.tpsa.regime_hmm_strategy import RegimeHmmStrategy
from app.tpsa.regime_hmm_risk_manager import RegimeHmmRiskManager

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
        kalman_filter_strategy = KalmanFilterStrategy(tickers, self.events_queue, self.initial_equity, ts0, ts1)
        regime_hmm_strategy = RegimeHmmStrategy(tickers, self.events_queue, 20000)
        #self.strategies = [kalman_filter_strategy, regime_hmm_strategy]
        #self.strategies = [kalman_filter_strategy]
        self.strategies = [regime_hmm_strategy]
        
        self.strategy = TpsaStrategy(
            tickers, self.events_queue, self.initial_equity, self.strategies
        )
        
        # Use the Naive Position Sizer where
        # suggested quantities are followed
        position_sizer = NaivePositionSizer()
        pickle_path = './work/hmm.pkl'
        hmm_model = pickle.load(open(pickle_path, "rb"))
        '''
        risk_managers = {
            'KalmanFilterStrategy': KalmanFilterRiskManager(hmm_model),
            'RegimeHmmStrategy': RegimeHmmRiskManager(hmm_model)
        }
        '''
        '''
        risk_managers = {
            'KalmanFilterStrategy': KalmanFilterRiskManager(hmm_model)
        }
        self.strategy.is_kalman_filter = True
        '''
        risk_managers = {
            'RegimeHmmStrategy': RegimeHmmRiskManager(hmm_model)
        }
        risk_manager = TpsaRiskManager(risk_managers=risk_managers)
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
        for si in self.strategies:
            si.portfolio = portfolio_handler.portfolio
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
        
        print('init_cash:{0}; equity:{1}; cur_cash:{2}; PnL:{3}'.format(
            portfolio_handler.portfolio.init_cash / PriceParser.PRICE_MULTIPLIER,
            portfolio_handler.portfolio.equity / PriceParser.PRICE_MULTIPLIER,
            portfolio_handler.portfolio.cur_cash / PriceParser.PRICE_MULTIPLIER,
            portfolio_handler.portfolio.realised_pnl / PriceParser.PRICE_MULTIPLIER
        ))
        print('持仓情况：')
        for key in portfolio_handler.portfolio.positions:
            item = portfolio_handler.portfolio.positions[key]
            print('ticker:{0}; quantity:{1};'.format(item.ticker, item.quantity))
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
