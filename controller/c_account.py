import datetime
import numpy as np
import tushare as ts
import pymysql
from app_registry import appRegistry as ar
from model.m_account import MAccount
from model.m_account_io import MAccountIo

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

    @staticmethod
    def withdraw(account_id, amount):
        '''
        从指定账户取出指定金额，增加资金转出金额记录，并更新用户现金资产
        '''
        return MAccountIo.withdraw(account_id, amount)

    @staticmethod
    def deposit(account_id, amount):
        '''
        向指定账户存一笔钱
        @param account_id：账户编号
        @param amount：存款数量
        @return 成功或失败
        @version v0.0.1 闫涛 2019-03-07
        '''
        return MAccountIo.deposit(account_id, amount)

    @staticmethod
    def update_stock_amount(account_id, stock_amount):
        '''
        更新用户的股票资产，值为用户持股量乘以前一个交易日收盘价，可以直接取用户持股表的数值
        @param account_id：账户编号
        @param stock_amount：股票资产值
        @version v0.0.1 闫涛 2019-03-07
        '''
        pk, affected_rows = MAccount.update_stock_amount(account_id, stock_amount)
        if 1 == affected_rows:
            return True
        else:
            return False
