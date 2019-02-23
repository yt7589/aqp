import numpy as np
from datetime import datetime
from datetime import timedelta
import copy
from model.m_stock_daily import MStockDaily

'''tpg
选择投资组合的仓位管理，首先由选股模块选择股票，本模块负责决定
各支股票占有的金额比例
'''
class CPortfolio(object):
    def __init__(self):
        self.name = 'CPortfolio'
        
    @staticmethod
    def get_portfolio(ts_codes, start_dt, duration):
        matrix_a = CPortfolio.get_matrix_a(ts_codes, start_dt, duration)
        cov = np.cov(matrix_a.T) # 求出协方差矩阵
        ans = np.linalg.eig(cov) # 求特值和特征向量
        print('协方差矩阵：')
        for row in cov:
            print('{0:#0.5f}, {1:#0.5f}, {2:#0.5f}, {3:#0.5f}, {4:#0.5f}'.format(row[0], row[1], row[2], row[3], row[4]))
        print('特征值和特征向量：')
        for i in range(len(ans[0])):
            print('特征值：{0}; 特征向量：{1}'.format(ans[0][i], ans[1][i]))
        ans_index = copy.copy(ans[0])
        ans_index.sort()
        print('排序后特征值：{0}'.format(ans_index))
        resu = []
        for k in range(len(ans_index)):
            con_temp = []
            con_temp.append(ans_index[k])
            content_temp1 = ans[1][ np.argwhere(ans[0] == ans_index[k])[0][0] ]
            content_temp1[content_temp1<=0] = 0
            sum = np.sum(content_temp1)
            content_temp1 /= sum
            con_temp.append(content_temp1)
            # 计算Sharp ratio
            sharp_temp = np.array(copy.copy(matrix_a)) * content_temp1
            sharp_exp = sharp_temp.mean()
            sharp_base = 0.0004
            sharp_std = np.std(sharp_temp)
            if sharp_std == 0.00:
                sharp = 0.00
            else:
                sharp = (sharp_exp - sharp_base) / sharp_std
            con_temp.append(sharp)
            resu.append(con_temp)
        for ii in resu:
            print(ii)
        print('v0.0.6')
        
    @staticmethod
    def get_matrix_a(ts_codes, start_dt, duration):
        '''
        获取股票组中股票每日收益率(cpt_chg)为矩阵的一行，取从指定日期
        开始start_time和duration指定天数为止的矩阵
        '''
        matrix_a = []
        start_dt = datetime(2018, 1, 1)
        for i in range(duration):
            curr_dt = start_dt + timedelta(days=i)
            next_dt = curr_dt + timedelta(days=1)
            rc, recs = MStockDaily.get_rate_of_returns(ts_codes, curr_dt.strftime('%Y%m%d'), next_dt.strftime('%Y%m%d'))
            if len(recs) >= 1:
                row = []
                for rec in recs:
                    row.append(rec[0])
                matrix_a.append(row)
        return np.array(matrix_a)
            