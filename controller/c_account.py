import datetime
import numpy as np
import tushare as ts
import pymysql
from app_registry import appRegistry as ar
from model.m_account import MAccount

'''
管理用户现金、股票等资产
'''
class CAccount(object):
    def __init__(self):
        self.name = 'CAccount'
        
    @staticmethod
    def get_current_amounts(account_id):
        '''
        获取用户当前现金资产值
        @author 闫涛 2019-02-15 v0.0.1
        '''
        rc, rows = MAccount.get_current_amounts(account_id)
        print('rows:{0} type:{1}'.format(rows, type(rows)))
        if rc > 0:
            return rows[0]
        else:
            return 0.0, 0.0

    @staticmethod
    def get_hist_amounts(account_id, account_date):
        '''
        获取指定日期用户账户资产，包括现金、股票资产等
        @author 闫涛 2019-03-01 v0.0.1
        '''
        rc, rows = MAccount.get_hist_amounts(account_id, account_date)
        print('rows:{0} type:{1}'.format(rows, type(rows)))
        if rc > 0:
            return rows[0]
        else:
            return 0.0, 0.0

    @staticmethod
    def update_cash_amount(account_id, cash_amount):
        pk, affected_rows = MAccount.update_cash_amount(account_id, cash_amount)
        if 1 == affected_rows:
            return True
        else:
            return False