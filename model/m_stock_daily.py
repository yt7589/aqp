import model.m_mysql as db

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