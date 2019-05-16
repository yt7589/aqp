#
from app.drl.holt_winters import HoltWinters
from app.drl.bitcoin_trading_engine import BitcoinTradingEngine

class DrlMain(object):
    def __init__(self):
        self.name = 'DrlMain'
        
    def startup(self):
        #holt_winters = HoltWinters()
        #holt_winters.startup()
        engine = BitcoinTradingEngine()
        engine.startup()
