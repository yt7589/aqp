# A股日线数据集类
from datetime import date
from datetime import timedelta
import numpy as np
from app_registry import appRegistry as ar
from controller.c_stock_daily import CStockDaily
from ann.stock_daily_svm import StockDailySvm

class AsdeDs(object):
    open_idx = 0
    high_idx = 1
    low_idx = 2
    close_idx = 3
    pre_close_idx = 4
    amt_chg_idx = 5
    pct_chg_idx = 6
    vol_idx = 7
    amount_idx = 8

    def __init__(self):
        self.name = 'AsdeDs'

    @staticmethod 
    def get_mean_stds(ds):
        ''' 求出训练样本集中开盘价、最高价、最低价、
        收盘价、前日收盘价、涨跌量、涨跌幅、交易量、金额的均值和标准差
        '''
        print('ds:{0}'.format(type(ds)))
        mus = np.zeros((10))
        stds = np.zeros((10))
        row_num = len(ds)
        # open
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.open_idx, row_num)
        # high
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.high_idx, row_num)
        # low
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.low_idx, row_num)
        # close
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.close_idx, row_num)
        # pre_close
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.pre_close_idx, row_num)
        # amt_chg
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.amt_chg_idx, row_num)
        # pct_chg
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.pct_chg_idx, row_num)
        # vol
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.vol_idx, row_num)
        # amount
        AsdeDs.get_mean_std(mus, stds, ds, AsdeDs.amount_idx, row_num)
        return mus, stds

    @staticmethod   
    def get_mean_std(mus, stds, x, idx, count):
        ''' 获取开盘价、最高价、最低价等单独列的均值和标准差 '''
        data = x[:, idx:idx+1].reshape(count)
        mus[idx] = np.mean(data)
        stds[idx] = np.std(data)
        
    @staticmethod
    def normalize_data(datas, idx, mus, stds):
        ''' 归一化方法：减去均值再除以标准差 '''
        datas[:, idx:idx+1] = (datas[:, idx:idx+1] - mus[idx]) / stds[idx]
        
    @staticmethod
    def normalize_datas(datas, mus, stds):
        ''' 对开盘价、最高价、最低价、收盘价等进行归一化 '''
        AsdeDs.normalize_data(datas, 
                    AsdeDs.open_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.high_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.low_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.close_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.pre_close_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.amt_chg_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.pct_chg_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.vol_idx,
                    mus, stds
        )
        AsdeDs.normalize_data(datas, 
                    AsdeDs.amount_idx,
                    mus, stds
        )

    