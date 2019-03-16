from model.m_user import MUser
#
class CUser(object):
    @staticmethod
    def get_user_id_of_account_id(account_id):
        '''
        获取账户编号所对应的用户编号
        @param account_id：int 账户编号
        @return int user_id用户编号
        @version v0.0.1 闫涛 2019-03-16
        '''
        rc, rows = MUser.get_user_id_of_account_id(account_id)
        if rc > 0:
            return rows[0][0]
        else:
            return 0