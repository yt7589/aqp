import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

class Chp022(object):
    def __init__(self):
        self.name = 'Chp022'
        # 数据文件格式：编号 日期 星期几 开盘价 最高价 
        # 最低价 收益价 收益
        self.data_file = 'data/pqb/chp022_001.txt'
        
    def startup(self):
        print('第22章：时间序列基本概念')
        # 读入指数数据
        index = pd.read_csv(self.data_file, sep='\t', index_col='Trddt')
        #self.get_data(index)
        sh_index = index[index.Indexcd==1]
        close_price = sh_index.Clsindex
        # self.draw_close_price(close_price)
        self.get_statistics_info(close_price)
        
        
    def get_data(self,  index):
        sh_index = index[index.Indexcd==1]
        print('上证综指类型：{0}'.format(type(sh_index)))
        print('上证综指前3条：{0}'.format(sh_index.head(n=3)))
        # 收盘价
        close_price = sh_index.Clsindex
        print('收盘价类型：{0}; 索引类型：{1}'.format(type(close_price), type(close_price.index)))
        print('收盘价前3条：{0}'.format(close_price.head(n=3)))
        close_price.index = pd.to_datetime(close_price.index)
        print('**收盘价类型：{0}; 索引类型：{1}'.format(type(close_price), type(close_price.index)))
        print('**收盘价前3条：{0}'.format(close_price.head(n=3)))
        
    def draw_close_price(self, close_price):
        xs = [d for d in close_price.index]
        font = FontProperties(fname='./work/simsun.ttc') # 载入中文字体
        plt.rcParams['axes.unicode_minus']=False # 正确显示负号
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=(mdates.MO)))
        plt.title('上证综指收盘价图', fontproperties=font)
        plt.xlabel('日期', fontproperties=font)
        plt.ylabel('指数' , fontproperties=font)
        plt.plot(xs, close_price)
        plt.gcf().autofmt_xdate() 
        plt.show()
        
    def get_statistics_info(self, close_price):
        print('最大值：{0}'.format(close_price.max()))
        print('最小值：{0}'.format(close_price.min()))
        print('平均值：{0}'.format(close_price.mean()))
        print('中位数：{0}'.format(close_price.median()))
        print('标准差：{0}'.format(close_price.std()))