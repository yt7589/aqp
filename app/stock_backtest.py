from datetime import date
from datetime import timedelta
import numpy as np
from app_registry import appRegistry as ar
from controller.c_stock_daily import CStockDaily
from ann.stock_daily_svm import StockDailySvm
from util.stock_daily_svm_model_evaluator import StockDailySvmModelEvaluator

class StockBacktest(object):
    def __init__(self):
        self.name = 'StockBacktest'

    def startup(self):
        print('股票回测研究平台 v0.0.1')
        tp = 0 # 预测上涨且实际上涨
        fp = 0 # 预测上涨但实际下跌
        tn = 0 # 预测下跌且实际下跌
        fn = 0 # 预测下跌但实际上涨
        # 选定股票
        stock_id = 69 # 603912.SH
        ts_code = '603912.SH'
        # 求出已知数据均值和方差
        start_dt = '20180101'
        end_dt = '20181231'
        mus, stds = StockDailySvmModelEvaluator.get_mean_stds(
                    ts_code, start_dt, end_dt
        )
        CStockDaily.generate_stock_daily_ds(ts_code, start_dt, end_dt)
        StockDailySvmModelEvaluator.normalize_datas(CStockDaily.train_x, mus, stds)        
        StockDailySvmModelEvaluator.normalize_datas(CStockDaily.test_x, mus, stds)
        # 将2019年第一个交易日作为当前数据作为测试数据
        current_date = date(2019, 1, 1)
        next_date = current_date + timedelta(days=1)
        rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, 
                    '{0}'.format(current_date), '{0}'.format(next_date))
        while rows is None or rc < 1:
            current_date = next_date
            next_date = current_date + timedelta(days=1)
            rc, rows = CStockDaily.get_stock_daily_from_db(ts_code, 
                        current_date.strftime('%Y%m%d'), next_date.strftime('%Y%m%d'))

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
            # 预测当前交易日涨跌
            rst = StockDailySvm.predict(CStockDaily.test_x)
            # 根据预测结果进行股票买卖
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
        


