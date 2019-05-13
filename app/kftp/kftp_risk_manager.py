# regime_hmm_risk_manager.py

from __future__ import print_function

import numpy as np

from qstrader.event import OrderEvent
from qstrader.price_parser import PriceParser
from qstrader.risk_manager.base import AbstractRiskManager
from app.kftp.regime_hmm_model import RegimeHmmModel


class KftpRiskManager(AbstractRiskManager):
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
    def __init__(self, hmm_model):
        self.name = 'KftpRiskManager'
        self.hmm_model = hmm_model

    def refine_orders(self, portfolio, sized_order):
        """
        Uses the Hidden Markov Model with the percentage returns
        to predict the current regime, either 0 for desirable or
        1 for undesirable. Long entry trades will only be carried
        out in regime 0, but closing trades are allowed in regime 1.
        """
        price_handler = portfolio.price_handler
        regime = RegimeHmmModel.determine_regime(
            self.hmm_model, price_handler, sized_order
        )
        action = sized_order.action
        # Create the order event, irrespective of the regime.
        # It will only be returned if the correct conditions 
        # are met below.
        order_event = OrderEvent(
            sized_order.ticker,
            sized_order.action,
            sized_order.quantity
        )
        return [order_event]