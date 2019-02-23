from datetime import date
from datetime import timedelta
import numpy as np
from app_registry import appRegistry as ar
from controller.c_stock_daily import CStockDaily
from ann.stock_daily_svm import StockDailySvm

class StockDailySvmModelEvaluator(object):
    # 1:open, 2:high, 3:low, 4:close, 5:pre_close, 
    # 6:amt_chg, 7:pct_chg, 8:vol, 9:amount
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
        self.name = 'StockDailySvmModelEvaluator'
        
    @staticmethod
    def evaluate_model(ts_code):
        print('评估模型好坏 v0.0.1')
        mus, stds = StockDailySvmModelEvaluator.get_mean_stds(
                    ts_code, '20180101', '29991231'
        )        
        tp = 0 # 预测上涨且实际上涨
        fp = 0 # 预测上涨但实际下跌
        tn = 0 # 预测下跌且实际下跌
        fn = 0 # 预测下跌但实际上涨
        # 取出2018年数据作为训练数据，
        CStockDaily.generate_stock_daily_ds(ts_code, '20180101', '20190101')
        StockDailySvmModelEvaluator.normalize_datas(CStockDaily.train_x, mus, stds)        
        StockDailySvmModelEvaluator.normalize_datas(CStockDaily.test_x, mus, stds)
        
        
        # 将2019年第一个交易日作为当前数据作为测试数据
        current_date = date(2019, 1, 1)
        next_date = current_date + timedelta(days=1)
        rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, '{0}'.format(current_date), '{0}'.format(next_date))
        
        while rows is None or rc < 1:
            current_date = next_date
            next_date = current_date + timedelta(days=1)
            rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, current_date.strftime('%Y%m%d'), next_date.strftime('%Y%m%d'))
        
        # 当前交易日不为空则循环执行
        while rc>0 and rows is not None:
            CStockDaily.train_x = np.append(CStockDaily.train_x, CStockDaily.test_x, axis=0)
            if CStockDaily.train_x[-1][3] / CStockDaily.train_x[-2][3] > ar.increase_threshold:
                CStockDaily.train_y = np.append(CStockDaily.train_y, 1)
            else:
                CStockDaily.train_y = np.append(CStockDaily.train_y, 0)
                        
            items = np.array(rows, dtype=np.float32)[:, 1:].reshape((1, 9))
            StockDailySvmModelEvaluator.normalize_datas(items, mus, stds)            
            CStockDaily.test_x = np.array([[
                float(items[0][0]), float(items[0][1]), 
                float(items[0][2]), float(items[0][3]), 
                float(items[0][4]), float(items[0][5]), 
                float(items[0][6]), int(items[0][7]), 
                float(items[0][8])
            ]])
            # 训练SVM模型
            StockDailySvm.train()
            hat = StockDailySvm.predict(CStockDaily.train_x)            
            
            # 预测当前交易日涨跌
            rst = StockDailySvm.predict(CStockDaily.test_x)
            print('result:{0} yesterday:{1} today:{2}'.format(rst, CStockDaily.train_x[-1, 3], CStockDaily.test_x[0, 3]))
            if rst[0] > 0:
                if CStockDaily.test_x[0][3] / CStockDaily.train_x[-1][3] > ar.increase_threshold:
                    tp += 1
                else:
                    fp += 1
            else:
                if CStockDaily.test_x[0][3] / CStockDaily.train_x[-1][3] > ar.increase_threshold:
                    fn += 1
                else:
                    tn += 1
            # 统计到tp,fp,tn,fn中
            # 将当前交易日数据添加到训练样本集
            # 将下一个交易日作为当前交易日
            next_date = current_date + timedelta(days=1)
            rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, '{0}'.format(current_date), '{0}'.format(next_date))
            
            while rows is None or rc < 1:
                current_date = next_date
                next_date = current_date + timedelta(days=1)
                rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, current_date.strftime('%Y%m%d'), next_date.strftime('%Y%m%d'))
                today = date.today()
                if next_date >= today:
                    break
            
        
        if tp==0 and fp==0 and fn==0:
            return
        
        # 计算准确率、召回率、F1值
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2*precision*recall) / (precision + recall)
        print('precision:{0}; recall:{1}; f1:{2} tp={3}; '
                    'fp={4}; fn={5}'.format(precision, 
                    recall, f1, tp, fp, fn))
        
    def get_mean_stds(ts_code, start_dt, end_dt):
        ''' 求出训练样本集中开盘价、最高价、最低价、
        收盘价、前日收盘价、涨跌量、涨跌幅、交易量、金额的均值和标准差
        '''
        CStockDaily.generate_stock_daily_ds(ts_code, start_dt, end_dt)
        mus = np.zeros((10))
        stds = np.zeros((10))
        row_num = len(CStockDaily.train_x)
        # open
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.open_idx, 
            row_num
        )
        print('v0.0.1: open:{0} {1}'.format(
                mus[StockDailySvmModelEvaluator.open_idx], 
                stds[StockDailySvmModelEvaluator.open_idx])
        )
        # high
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.high_idx, 
            row_num
        )
        print('high: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.high_idx],
                    stds[StockDailySvmModelEvaluator.high_idx])
        )
        # low
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.low_idx, 
            row_num
        )
        print('low: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.low_idx],
                    stds[StockDailySvmModelEvaluator.low_idx])
        )
        # close
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.close_idx, 
            row_num
        )
        print('close: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.close_idx],
                    stds[StockDailySvmModelEvaluator.close_idx])
        )
        # pre_close
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.pre_close_idx, 
            row_num
        )
        print('pre_close: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.pre_close_idx],
                    stds[StockDailySvmModelEvaluator.pre_close_idx])
        )
        # amt_chg
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.amt_chg_idx, 
            row_num
        )
        print('amt_chg: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.amt_chg_idx],
                    stds[StockDailySvmModelEvaluator.amt_chg_idx])
        )
        # pct_chg
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.pct_chg_idx, 
            row_num
        )
        print('pct_chg: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.pct_chg_idx],
                    stds[StockDailySvmModelEvaluator.pct_chg_idx])
        )
        # vol
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.vol_idx, 
            row_num
        )
        print('vol: {0}, {1}'.format(mus[StockDailySvmModelEvaluator.vol_idx],
                    stds[StockDailySvmModelEvaluator.vol_idx])
        )
        # amount
        StockDailySvmModelEvaluator.get_mean_std(
            mus, stds,
            CStockDaily.train_x, 
            StockDailySvmModelEvaluator.amount_idx, 
            row_num
        )
        print('amount: {0}, {1}'.format(
                    mus[StockDailySvmModelEvaluator.amount_idx],
                    stds[StockDailySvmModelEvaluator.amount_idx])
        )
        return mus, stds
        
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
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.open_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.high_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.low_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.close_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.pre_close_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.amt_chg_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.pct_chg_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.vol_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.amount_idx,
                    mus, stds
        )