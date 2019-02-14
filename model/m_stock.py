import model.m_mysql as db

class MStock(object):
    def __init__(self):
        self.name = 'MStock'
        
    @staticmethod
    def get_stock_id_by_name(stock_name):
        sql = 'select stock_id from t_stock where stock_name=%s'
        params = (stock_name)
        rowcount, rows = db.query(sql, params)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return -1
            
    @staticmethod
    def get_stock_ts_code_by_id(stock_id):
        sql = 'select ts_code from t_stock where stock_id=%d'
        params = (stock_id)
        rowcount, rows = db.query(sql, params)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return ''
            
    @staticmethod
    def add_stock(ts_code, symbol, stock_name, area_id, industry_id, list_date):
        sql = 'insert into t_stock(ts_code, symbol, stock_name, area_id, industry_id, list_date) values(%s, %s, %s, %s, %s, %s)'
        params = (ts_code, symbol, stock_name, area_id, industry_id, list_date)
        industry_id, affected_rows = db.insert(sql, params)
        return industry_id