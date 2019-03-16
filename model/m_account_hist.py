import model.m_mysql as db
from util.app_util import AppUtil

'''
账户历史信息表，记录账户每日现金资产和股票资产，以前一交易日收盘价
计算总现金资产
'''
class MAccountHist(object):
    @staticmethod
    def insert_account_hist(account_id, account_date, 
                cash_amount, stock_amount):
        '''
        向t_account_hist表中添加一条记录，记录用户某天的资产总值
        @param account_id：int 账户编号
        @param account_date：datetime 账务日期
        @param cash_amount：int 以分为单位的现金资产
        @param stock_amount：int 以前一交易日收盘价计算的股票资产
        @return 新插入的主键和插入行数
        @version v0.0.1 闫涛 2019-03-16
        '''
        sql = 'insert into t_account_hist(account_id, '\
                    'account_date, cash_amount, stock_amount) '\
                    'values(%s, %s, %s, %s)'
        params = (account_id, AppUtil.format_date(account_date, \
                    AppUtil.DF_HYPHEN), cash_amount, stock_amount)
        return db.insert(sql, params)