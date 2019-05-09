import csv
import matplotlib.pyplot as plt
from core.quotation.bs_cna_daily import BsCnaDaily
# 
from app.kftp.kftp_dataset import KftpDataset
#from app.tpsa.tpsa_engine import TpsaEngine
#from app.tpsa.regime_hmm_model import RegimeHmmModel
#from app.tpsa.regime_hmm_engine import RegimeHmmEngine

class KftpMain(object):
    def __init__(self):
        self.name = 'KftpMain'
        
    def startup(self):
        print('卡尔曼滤波交易对策略')
        #self.get_quotation_data() # 获取行情数据
        # 绘制收盘价曲线
        stock_files = ['./data/ICBC.csv', './data/CBC.csv']
        KftpDataset.draw_close_price_curve(stock_files)
        # 运行卡尔曼滤波模型
        #tpsaEngine = TpsaEngine()
        #tpsaEngine.startup()
        
        # 训练隐马可夫模型（因为隐马可夫模型随机给定状态值，
        # 有时会是状态0适合交易，有时会是状态1适合交易，
        # 预训练可以固定为状态0还是状态1适合交易）
        #rhm = RegimeHmmModel()
        #rhm.train()
        # 实际运行隐马可夫模型
        #rhe = RegimeHmmEngine()
        #rhe.startup()
        
    def get_quotation_data(self):
        stocks = [
            {'stock_code': 'sh.601398', 'start_date': '2017-01-01', 'end_date': '2019-04-26', 'etf_name': 'ICBC'},
            {'stock_code': 'sh.601939', 'start_date': '2017-01-01', 'end_date': '2019-04-26', 'etf_name': 'CBC'}
        ]
        KftpDataset.get_quotation_data(stocks)