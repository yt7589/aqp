import datetime
import numpy as np
import tushare as ts
import pymysql
from app_registry import appRegistry as ar
from model.m_stock_daily import MStockDaily

'''
获取沪深两市指定股票（股票代码）的日K线数据
'''
class CStockDaily(object):
    def __init__(self):
        self.name = 'CStockDaily'
        
    @staticmethod
    def get_stock_daily_kline(ts_code, start_dt, end_dt):
        '''
        获取指定股票的日K线数据
        stock_id：股票编号
        start_dt: 开始日期 20190215
        end_dt：结束日期 20190215
        @return K线数据：日期、开盘价、收盘价、最高价、最低价、
                    交易量、交易金额、涨跌值、涨跌幅
        @author 闫涛 2019-02-15 v0.0.1
        '''
        ts.set_token(ar.ts_token)
        pro = ts.pro_api()
        df = pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)
        rec_nums = df.shape[0]
        for i in range(rec_nums):
            rec = list(df.ix[rec_nums-1-i])
            stock_daily_vo = {
                'state_dt': rec[1],
                'stock_code': rec[0],
                'open': float(rec[2]),
                'high': float(rec[3]),
                'low': float(rec[4]),
                'close': float(rec[5]),
                'pre_close': float(rec[6]),
                'amt_chg': float(rec[7]),
                'pct_chg': float(rec[8]),
                'vol': int(rec[9]),
                'amount': float(rec[10])
            }
            MStockDaily.add_stock_daily(stock_daily_vo)
            print('{0} | {1} | {2} | {3}'.format(rec[0], 
                        rec[1], rec[2], rec[3]))
        print('OK?????????')
        
    @staticmethod
    def generate_stock_daily_ds(stock_code, start_dt, end_dt):
        rc, rows = MStockDaily.get_stock_daily(stock_code, start_dt, end_dt)
        train_x = []
        train_y = []
        for i in range(rc - 1):
            train_x.append([float(rows[i][1]), float(rows[i][2]), 
                        float(rows[i][3]), float(rows[i][4]), 
                        float(rows[i][5]), float(rows[i][6]), 
                        float(rows[i][7]), int(rows[i][8]), 
                        float(rows[i][9])])
            if i>=1 and rows[i][3] / rows[i-1][3] > ar.increase_threshold:
                train_y.append(1)
            else:
                train_y.append(0)
        CStockDaily.train_x = np.array(train_x)
        CStockDaily.train_y = np.array(train_y)
        v1 = np.array(train_y)
        test_x = [[
            float(rows[rc-1][1]), 
            float(rows[rc-1][2]), float(rows[rc-1][3]), 
            float(rows[rc-1][4]), float(rows[rc-1][5]), 
            float(rows[rc-1][6]), float(rows[rc-1][7]), 
            int(rows[rc-1][8]), float(rows[rc-1][9])
        ]]
        CStockDaily.validate_x = np.array([])
        CStockDaily.validate_y = np.array([])
        CStockDaily.test_x = np.array(test_x)
        
    @staticmethod
    def get_stock_daily_from_db(ts_code, start_dt, end_dt):
        return MStockDaily.get_stock_daily(ts_code, start_dt, end_dt)
        
        
        
        
        
        
        
        
        
        
        
        
        
        