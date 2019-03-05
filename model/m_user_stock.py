import model.m_mysql as db

'''
管理用户持有股票的信息
'''
class MUserStock(object):
    def __init__(self):
        self.name = 'MUserStock'

    @staticmethod
    def get_user_stock_id(user_id, stock_id):
        '''
        根据用户编号和股票编号求出user_stock_id
        @param user_id：用户编号
        @param stock_id：股票编号
        @return user_stock_id
        @version v0.0.1 闫涛 2019-03-04
        '''
        sql = 'select user_stock_id from t_user_stock where user_id=%s and stock_id=%s'
        params = (user_id, stock_id)
        return db.query(sql, params)

    @staticmethod
    def get_user_stock_hold(user_stock_id):
        '''
        获取用户指定股票的持有量
        @param user_stock_id：用户持有某种股票的编号
        @version v0.0.1 闫涛 2019-03-04
        '''
        sql = 'select hold_vol from t_user_stock where user_stock_id=%s'
        params = (user_stock_id)
        return db.query(sql, params)

    @staticmethod
    def get_user_and_stock_hold(user_id, stock_id):
        '''
        '''
        user_stock_id = MUserStock.get_user_stock_id(user_id, stock_id)
        return MUserStock.get_user_stock_hold(user_stock_id)

    @staticmethod
    def get_stock_vo(user_stock_id):
        '''
        通过用户持股编号查询股票基本信息
        @param user_stock_id：用户持股编号
        @return 股票值对象，包括股票编号和代码、名称等
        @version v0.0.1 闫涛 2019-03-05
        '''
        sql = 'select S.stock_id, S.ts_code, S.symbol, '\
                'S.stock_name from t_user_stock US, t_stock S '\
                'where US.stock_id=S.stock_id and US.user_stock_id=%s'
        params = (user_stock_id)
        return db.query(sql, params)