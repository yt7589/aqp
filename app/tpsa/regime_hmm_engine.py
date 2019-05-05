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
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.price_handler.bscna_daily_csv_bar import BscnaDailyCsvBarPriceHandler

import matplotlib.pyplot as plt
from app.tpsa.tpsa_dataset import TpsaDataset

from app.tpsa.regime_hmm_strategy import RegimeHmmStrategy
from app.tpsa.regime_hmm_model import RegimeHmmModel
from app.tpsa.regime_hmm_risk_manager import RegimeHmmRiskManager
from app.tpsa.user_account import UserAccount

class RegimeHmmEngine(object):
    def __init__(self):
        self.name = 'RegimeHmmEngine'
        
    def startup(self):
        testing = False
        config = settings.load_config()
        tickers = ["ICBC"]
        filename = None
        self.run(config, testing, tickers, filename)
        
    def run(self, config, testing, tickers, filename):
        '''
        '''
        
        title = ['隐马可夫模型CPA策略']
        pickle_path = './work/hmm.pkl'
        csv_dir = './data/'
        events_queue = queue.Queue()
        initial_equity = 500000.00 # 初始资金
        start_date = datetime.datetime(2017, 1, 1)
        end_date = datetime.datetime(2019, 4, 26)
        base_quantity = 10000
        
        # 用户账户
        user_account = UserAccount(initial_equity, tickers)
        
        
        
        
        
        
        strategy = RegimeHmmStrategy(
            tickers, events_queue, base_quantity,
            short_window=10, long_window=30
        )
        price_handler = BscnaDailyCsvBarPriceHandler(
            csv_dir, events_queue, tickers,
            start_date=start_date, 
            end_date=end_date,
            calc_adj_returns=True
        )
        position_sizer = NaivePositionSizer()
        hmm_model = pickle.load(open(pickle_path, "rb"))
        #regime_hmm_model = RegimeHmmModel()
        #hmm_model = regime_hmm_model.train()
        risk_manager = RegimeHmmRiskManager(hmm_model)
        portfolio_handler = PortfolioHandler(
            PriceParser.parse(initial_equity), 
            events_queue, price_handler,
            position_sizer, risk_manager
        )
        statistics = TearsheetStatistics(
            config, portfolio_handler, 
            title, benchmark="ICBC"
        )
        backtest = TradingSession(
            config, strategy, tickers,
            initial_equity, start_date, end_date,
            events_queue, title=title,
            price_handler=price_handler,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            statistics=statistics,
            portfolio_handler=portfolio_handler
        )
        results = backtest.start_trading(testing=testing)
        return results
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
