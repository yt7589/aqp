import time
import datetime
from datetime import date
import model.m_mysql as db
from util.app_util import AppUtil

'''
股票日行情信息模型类，负责t_stock_daily表的增、删、改操作
'''
class MStockDaily(object):
    def __init__(self):
        self.name = 'MStock'
        
    @staticmethod
    def add_stock_daily(vo):
        '''
        添加股票日行情数据
        '''
        sql = 'insert into t_stock_daily(state_dt, stock_code, open, high, low, close, '\
                'pre_close, amt_chg, pct_chg, vol, amount)'\
                ' values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
        params = (vo['state_dt'], vo['stock_code'], vo['open'], vo['high'], 
                    vo['low'], vo['close'], vo['pre_close'], vo['amt_chg'], 
                    vo['pct_chg'], vo['vol'], vo['amount'])
        stock_daily_id, affected_rows = db.insert(sql, params)
        return stock_daily_id
        
    @staticmethod
    def get_stock_daily(stock_code, start_dt, end_dt):
        sql = 'select state_dt, open, high, low, close, pre_close, '\
                    'amt_chg, pct_chg, vol, amount from t_stock_daily'\
                    ' where stock_code=%s '\
                    'and state_dt>=%s and state_dt<%s'
        params = (stock_code, start_dt, end_dt)
        return db.query(sql, params)
        
    @staticmethod
    def get_rate_of_returns(ts_codes, start_dt, end_dt):
        stocks = '\',\''.join(ts_codes)
        stock_codes = '\'{0}\''.format(stocks)
        sql = 'select pct_chg/100.0 from t_stock_daily where stock_code in ({0}) and state_dt>=%s and state_dt<%s order by stock_code'.format(stock_codes)
        params = (start_dt, end_dt)
        return db.query(sql, params)

    @staticmethod
    def get_close(ts_code, dt):
        '''
        获取指定股票在指定日期的收盘价
        '''
        sql = 'select close from t_stock_daily where '\
                    'stock_code=%s and state_dt>=%s and state_dt<%s'
        curr_date = dt
        next_date = AppUtil.get_delta_date(dt, 1)
        params = (ts_code, curr_date, next_date)
        return db.query(sql, params)