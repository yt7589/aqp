import csv
import matplotlib.pyplot as plt
from core.quotation.bs_cna_daily import BsCnaDaily
from app.tpsa.tpsa_engine import TpsaEngine

from app.tpsa.tpsa_dataset import TpsaDataset
from app.tpsa.regime_hmm_engine import RegimeHmmEngine

class TpsaMain(object):
    def __init__(self):
        self.name = 'TpsaMain'
        
    def startup(self):
        print('A股交易对策略')
        #self.get_quotation_data() # 获取行情数据
        #self.draw_close_price_curve()
        #tpsaEngine = TpsaEngine()
        #tpsaEngine.startup()
        
        #stock_files = ['./data/ICBC.csv', './data/CBC.csv']
        #TpsaDataset.draw_close_price_curve(stock_files)
        
        #rhm = RegimeHmmModel()
        #rhm.train()
        rhe = RegimeHmmEngine()
        rhe.startup()