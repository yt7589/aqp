import csv
import matplotlib.pyplot as plt
from core.quotation.bs_cna_daily import BsCnaDaily
from app.tpsa.tpsa_engine import TpsaEngine

from app.tpsa.tpsa_dataset import TpsaDataset
from app.tpsa.regime_hmm_model import RegimeHmmModel
from app.tpsa.regime_hmm_engine import RegimeHmmEngine
<<<<<<< HEAD
=======
from app.tpsa.user_account import UserAccount
>>>>>>> 26adae1a1a30b832837260b6e49d978e9323631a

class TpsaMain(object):
    def __init__(self):
        self.name = 'TpsaMain'
        
    def startup(self):
        print('A股交易对策略')
        #self.get_quotation_data() # 获取行情数据
        #self.draw_close_price_curve()
        # 绘制收盘价曲线
        #stock_files = ['./data/ICBC.csv', './data/CBC.csv']
        #TpsaDataset.draw_close_price_curve(stock_files)
        # 运行卡尔曼滤波模型
        #tpsaEngine = TpsaEngine()
        #tpsaEngine.startup()
        
        # 训练隐马可夫模型（因为隐马可夫模型随机给定状态值，
        # 有时会是状态0适合交易，有时会是状态1适合交易，
        # 预训练可以固定为状态0还是状态1适合交易）
        #rhm = RegimeHmmModel()
        #rhm.train()
        # 实际运行隐马可夫模型
        rhe = RegimeHmmEngine()
        rhe.startup()