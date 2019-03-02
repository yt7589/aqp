import model.m_mysql as db

class MAccountIo(object):
    def __init__(self):
        self.name = 'MAccountIo'

    @staticmethod
    def get_latest_io(account_id):
        '''
        找出指定账户最新一条流水记录
        @version v0.0.1 闫涛 2019.03.02
        '''
        sql = 'select balance from t_account_io where account_id=%s'
        params = (account_id)
        return db.query(sql, params)