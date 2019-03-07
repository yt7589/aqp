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

    @staticmethod
    def update_cash_amount(account_id, cash_amount):
        '''
        更新账户现金资产数值
        @param account_id：账户
        @param cash_amount：新的现金资产数
        @version v0.0.1 闫涛 2019-03-04
        '''
        sql = 'update t_account set cash_amount=%s where account_id=%s'
        params = (cash_amount, account_id)
        return db.update(sql, params)

    @staticmethod
    def update_stock_amount(account_id, stock_amount):
        '''
        更新用户的股票资产，值为用户持股量乘以前一个交易日收盘价，可以直接取用户持股表的数值
        @param account_id：账户编号
        @param stock_amount：股票资产值
        @version v0.0.1 闫涛 2019-03-07
        '''
        sql = 'update t_account set stock_amount=%s where account_id=%s'
        params = (stock_amount, account_id)
        return db.update(sql, params)