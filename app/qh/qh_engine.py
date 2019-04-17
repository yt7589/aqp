import calendar
import datetime

from qstrader import settings
from qstrader.strategy.base import AbstractStrategy
from qstrader.position_sizer.rebalance import LiquidateRebalancePositionSizer
from qstrader.event import SignalEvent, EventType
from qstrader.compat import queue
from qstrader.trading_session import TradingSession

import matplotlib.pyplot as plt

from app.qh.qh_strategy import QhStrategy

class QhEngine(object):
    def __init__(self):
        self.name = 'QhEngine'
        
    def startup(self):
        testing = False
        config = settings.load_config()
        tickers = ["SPY", "AGG"]
        self.run(config, testing, tickers) 

    def run(self, config, testing, tickers):
        # Backtest information
        title = [
            'Monthly Liquidate/Rebalance on 60%/40% SPY/AGG Portfolio'
        ]
        initial_equity = 500000.0
        start_date = datetime.datetime(2006, 11, 1)
        end_date = datetime.datetime(2016, 10, 12)

        # Use the Monthly Liquidate And Rebalance strategy
        events_queue = queue.Queue()
        strategy = QhStrategy(
            tickers, events_queue
        )

        # Use the liquidate and rebalance position sizer
        # with prespecified ticker weights
        ticker_weights = {
            "SPY": 0.6,
            "AGG": 0.4,
        }
        position_sizer = LiquidateRebalancePositionSizer(
            ticker_weights
        )

        # Set up the backtest
        backtest = TradingSession(
            config, strategy, tickers,
            initial_equity, start_date, end_date,
            events_queue, position_sizer=position_sizer,
            title=title, benchmark=tickers[0],
        )
        results = backtest.start_trading(testing=testing)
        plt.show()
        return results
