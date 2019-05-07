import csv
import pandas as pd
import matplotlib.pyplot as plt
from core.quotation.bs_cna_daily import BsCnaDaily

class KftpDataset(object):
    def __init__(self):
        self.name = 'TpsaDataset'
        
    @staticmethod
    def get_quotation_data(stocks):
        '''
        获取交易对行情数据
        @params stocks np.array [0][...]和[1][...]分别代表交易对
            对于交易对：[0]{stock_code, start_date, end_date, eft_name}
            其中eft_name也是数据文件保存的名字，stocks共有两条记录，由
            调用者负责保证
        '''
        print(stocks[0]['stock_code'])
        bs_cna_daily = BsCnaDaily()
        # 工商银行（股票代码60开头的是上海）
        bs_cna_daily.get_history_data(stocks[0]['stock_code'], 
                    stocks[0]['start_date'], stocks[0]['end_date'], 
                    stocks[0]['etf_name'])
        bs_cna_daily.get_history_data(stocks[1]['stock_code'], 
                    stocks[1]['start_date'], stocks[1]['end_date'], 
                    stocks[1]['etf_name'])
            
    @staticmethod
    def draw_close_price_curve(stock_files):
        '''
        绘制股票收益率曲线
        @stock_files 交易对股票行情文件名，共有两个
        '''
        print('绘制收盘价曲线...')
        etf0_prices = TpsaDataset.read_close_prices(stock_files[0])
        etf1_prices = TpsaDataset.read_close_prices(stock_files[1])
        plt.title('close price curve')
        plt.plot(etf0_prices)
        plt.plot(etf1_prices)
        plt.show()
    
    @staticmethod
    def read_close_prices(stock_file):
        '''
        读取股票收盘价数据时间序列
        @param stock_file 股票行情文件
        @return 股票收盘价的list
        '''
        close_prices = []
        with open(stock_file, 'r', newline='') as fd:
            rows = csv.reader(fd)
            header = next(rows)
            print('header:{0}'.format(header))
            for row in rows:
                close_prices.append(float(row[4]))
        return close_prices
        
    @staticmethod
    def read_close_price_pd(stock_file):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        prices = pd.read_csv(stock_file, encoding='utf-8', 
                parse_dates=['Date'], date_parser=dateparse, 
                index_col='Date')
        print(prices.values[:, 3])
        
        
        
        
        
        
        
        