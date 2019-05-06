import csv
import matplotlib.pyplot as plt
from core.quotation.bs_cna_daily import BsCnaDaily
from app.hrmc.hrmc_dataset import HrmcDataset
from app.hrmc.hrmc_hmm_model import HrmcHmmModel
from app.hrmc.hrmc_engine import HrmcEngine

class HrmcMain(object):
    def __init__(self):
        self.name = 'HrmcMain'
        
    def startup(self):
        print('A股交易对策略')
        # 获取行情文件
        '''
        stocks = [
            {
                'stock_code':'sh.601398', 
                'start_date': '2008-01-01', 
                'end_date': '2019-04-26', 
                'etf_name': 'ICBC'
            }
        ]
        HrmcDataset.get_quotation_data(stocks)
        # 绘制收盘价曲线
        stock_files = ['./data/ICBC.csv']
        HrmcDataset.draw_close_price_curve(stock_files)
        '''
        
        
        
        #self.get_quotation_data() # 获取行情数据
        #self.draw_close_price_curve()
        # 运行卡尔曼滤波模型
        
        # 训练隐马可夫模型（因为隐马可夫模型随机给定状态值，
        # 有时会是状态0适合交易，有时会是状态1适合交易，
        # 预训练可以固定为状态0还是状态1适合交易）
        #rhm = HrmcHmmModel()
        #rhm.train()
        
        # 运行模型
        hrmcEngine = HrmcEngine()
        hrmcEngine.startup()