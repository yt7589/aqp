import model.m_mysql as db

class MUser(object):
    @staticmethod
    def get_user_id_of_account_id(account_id):
        '''
        获取账户编号所对应的用户编号
        @param account_id：int 账户编号
        @return int user_id用户编号
        @version v0.0.1 闫涛 2019-03-16
        '''
        sql = 'select user_id from t_user where account_id=%s'
        params = (account_id)
        return db.query(sql, params)