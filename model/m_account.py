import time
import datetime
from datetime import date
import model.m_mysql as db

'''
账户模型类，负责t_account表的增、删、改操作
'''
class MAccount(object):
    def __init__(self):
        self.name = 'MAccount'

    @staticmethod
    def get_current_amounts(account_id):
        '''
        获取现金资产金额，以分为单位
        '''
        sql = 'select cash_amount, stock_amount from t_account where account_id=%s'
        params = (account_id)
        return db.query(sql, params)

    @staticmethod
    def get_hist_amounts(account_id, account_date):
        sql = 'select cash_amount, stock_amount from t_account_hist '\
                        'where account_id=%s and account_date=%s'
        params = (account_id, account_date)
        return db.query(sql, params)