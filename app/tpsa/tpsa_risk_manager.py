# regime_hmm_risk_manager.py

from __future__ import print_function

import numpy as np

from qstrader.event import OrderEvent
from qstrader.price_parser import PriceParser
from qstrader.risk_manager.base import AbstractRiskManager


class TpsaRiskManager(AbstractRiskManager):
    """
    Utilises a previously fitted Hidden Markov Model 
    as a regime detection mechanism. The risk manager
    ignores orders that occur during a non-desirable
    regime.

    It also accounts for the fact that a trade may
    straddle two separate regimes. If a close order
    is received in the undesirable regime, and the 
    order is open, it will be closed, but no new
    orders are generated until the desirable regime
    is achieved.
    """
    def __init__(self):
        self.name = 'TpsaRiskManager'
        self.risk_managers = {}
        
    def register_risk_manager(strategy_name, risk_manager_obj):
        self.risk_managers[strategy_name] = risk_manager_obj

    def refine_orders(self, portfolio, sized_order):
        """
        Uses the Hidden Markov Model with the percentage returns
        to predict the current regime, either 0 for desirable or
        1 for undesirable. Long entry trades will only be carried
        out in regime 0, but closing trades are allowed in regime 1.
        """
        print('风控模块运行了，检查风险中...{0} ########'.format(sized_order.strategy_name))
        if sized_order.strategy_name in self.risk_managers:
            return self.risk_managers[sized_order.strategy_name].refine_orders(portfolio, sized_order)
        # Create the order event, irrespective of the regime.
        # It will only be returned if the correct conditions 
        # are met below.
        order_event = OrderEvent(
            sized_order.ticker,
            sized_order.action,
            sized_order.quantity
        )
        return [order_event]