import model.m_mysql as db

'''
股票信息模型类，负责t_stock表的增、删、改操作
'''
class MStock(object):
    def __init__(self):
        self.name = 'MStock'
        
    @staticmethod
    def get_stock_id_by_name(stock_name):
        ''' 根据股票名称查询股票编号 '''
        sql = 'select stock_id from t_stock where stock_name=%s'
        params = (stock_name)
        rowcount, rows = db.query(sql, params)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return -1
            
    @staticmethod
    def get_stock_ts_code_by_id(stock_id):
        ''' 根据股票编号查询股票代码 '''
        sql = 'select ts_code from t_stock where stock_id=%d'
        params = (stock_id)
        rowcount, rows = db.query(sql, params)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return ''
            
    @staticmethod
    def add_stock(ts_code, symbol, stock_name, area_id, 
                industry_id, list_date):
        ''' 添加股票基本信息 '''
        sql = 'insert into t_stock(ts_code, symbol, stock_name, '\
                'area_id, industry_id, list_date) values(%s, '\
                '%s, %s, %s, %s, %s)'
        params = (ts_code, symbol, stock_name, area_id, 
                    industry_id, list_date)
        industry_id, affected_rows = db.insert(sql, params)
        return industry_id

    @staticmethod
    def get_stock_vo_by_id(stock_id):
        '''
        根据股票编号查询股票基本信息
        @param stock_id；股票编号
        @version v0.0.1 闫涛 2019-03-06
        '''
        sql = 'select ts_code, symbol, stock_name, area_id, industry_id'\
                    ' from t_stock where stock_id=%s'
        params = (stock_id)
        return db.query(sql, params)

    @staticmethod
    def get_stock_id_by_ts_code(ts_code):
        '''
        根据ts_code求出股票编号stock_id
        @param ts_code：股票编码
        @return stock_id
        @version v0.0.1 闫涛 2019-03-06
        '''
        sql = 'select stock_id from t_stock where ts_code=%s'
        params = (ts_code)
        return db.query(sql, params)