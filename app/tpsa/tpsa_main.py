import csv
import matplotlib.pyplot as plt
from core.quotation.bs_cna_daily import BsCnaDaily
from app.tpsa.tpsa_engine import TpsaEngine

class TpsaMain(object):
    def __init__(self):
        self.name = 'TpsaMain'
        
    def startup(self):
        print('A股交易对策略')
        #self.get_quotation_data() # 获取行情数据
        #tpsaEngine = TpsaEngine()
        #tpsaEngine.startup()
        self.draw_close_price_curve()
        
    def get_quotation_data(self):
        # 取工商银行2016-10-27~2019-04-23
        bs_cna_daily = BsCnaDaily()
        # 工商银行（股票代码60开头的是上海）
        bs_cna_daily.get_history_data('sh.601398', 
                    '2017-01-01', '2019-04-23', 'ICBC')
        # 建设银行
        bs_cna_daily.get_history_data('sh.601939', 
                    '2017-01-01', '2019-04-23', 'CBC')
        # 浦发银行
        bs_cna_daily.get_history_data('sh.600000', 
                    '2017-01-01', '2019-04-23', 'PF')
                    
    def draw_close_price_curve(self):
        print('绘制收盘价曲线...')
        icbc_prices = self.read_close_prices('./data/ICBC.csv')
        cbc_prices = self.read_close_prices('./data/CBC.csv')
        plt.plot(icbc_prices)
        plt.plot(cbc_prices)
        plt.show()
        
    def read_close_prices(self, stock_file):
        close_prices = []
        with open(stock_file, 'r', newline='') as fd:
            rows = csv.reader(fd)
            header = next(rows)
            print('header:{0}'.format(header))
            for row in rows:
                close_prices.append(float(row[4]))
        return close_prices