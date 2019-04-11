import datetime
from datetime import timedelta
import numpy as np
import tushare as ts
from app_registry import appRegistry as ar
from model.m_stock_daily import MStockDaily
from util.app_util import AppUtil

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
        '''
        求出并返回指定股票在指定时间段的训练样本集、验证样本集、试验样本集
        @param stock_code：股票编码
        @param start_dt：开始时间
        @param end_dt：结束时间
        @return 训练样本集、验证样本集、测试样本集
        @author v0.0.2 闫涛 2019-03-12 作适用于策略引擎的修改
        '''
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
        validate_x = np.array([])
        validate_y = np.array([])
        test_x = [[
            float(rows[rc-1][1]), 
            float(rows[rc-1][2]), float(rows[rc-1][3]), 
            float(rows[rc-1][4]), float(rows[rc-1][5]), 
            float(rows[rc-1][6]), float(rows[rc-1][7]), 
            int(rows[rc-1][8]), float(rows[rc-1][9])
        ]]
        CStockDaily.validate_x = validate_x
        CStockDaily.validate_y = validate_y
        CStockDaily.test_x = np.array(test_x)
        return np.array(train_x), np.array(train_y), \
                validate_x, validate_y, \
                np.array(test_x)
        
    @staticmethod
    def get_stock_daily_from_db(ts_code, start_dt, end_dt):
        return MStockDaily.get_stock_daily(ts_code, start_dt, end_dt)

    MODE_CLOSE_QUOTATION = 1001
    MODE_TRADE_QUOTATION = 1002
    @staticmethod
    def get_daily_quotation(ts_code, ask_date, mode):
        '''
        求出ask_date的行情数据，如果ask_date不是交易日，就取ask_date的下一个
        交易日
        @param ts_code：股票编码
        @param ask_date：要查询的日期
        @return 交易日（date类型）和行情数据
        '''
        if CStockDaily.MODE_CLOSE_QUOTATION == mode:
            delta = -1
        else:
            delta = 1
        next_date = ask_date + timedelta(days=delta)
        print('next_date:{0}'.format(AppUtil.format_date(next_date)))
        rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, 
                        AppUtil.format_date(ask_date), 
                        AppUtil.format_date(next_date))
        while rc < 1 or rows is None:
            ask_date = next_date
            next_date = ask_date + timedelta(days=delta)
            sql_date = ask_date + timedelta(days=1)
            rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, 
                        AppUtil.format_date(ask_date), 
                        AppUtil.format_date(sql_date))
        print('rows;{0}'.format(rows[0]))
        quotation = []
        quotation.append(float(rows[0][1])) # open
        quotation.append(float(rows[0][2]))
        quotation.append(float(rows[0][3]))
        quotation.append(float(rows[0][4]))
        quotation.append(float(rows[0][5]))
        quotation.append(float(rows[0][6]))
        quotation.append(float(rows[0][7]))
        quotation.append(float(rows[0][8]))
        quotation.append(float(rows[0][9]))
        return ask_date, np.array(quotation)

    @staticmethod
    def get_close(ts_code, dt):
        '''
        获取指定股票在指定日期的收盘价
        '''
        rc, recs = MStockDaily.get_close(ts_code, dt)
        if rc > 0:
            return recs[0][0]
        else:
            return -1.0

    @staticmethod
    def get_real_close(ts_code, dt):
        '''
        获取指定日期收盘价，若当天没有收盘价，则取前一交易日的收盘价
        @param ts_code：股票编码
        @param dt：日期
        @return 当前或前一交易日的收盘价
        @version v0.0.1 闫涛 2019-03-05
        '''
        rc, rows = MStockDaily.get_close(ts_code, dt)
        while rc <= 0:
            prev_date = AppUtil.get_delta_date(dt, -1)
            dt = prev_date
            rc, rows = MStockDaily.get_close(ts_code, prev_date)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return -1.0

        
        
        
        
        
        
        
        
        
        
        
        
        
        