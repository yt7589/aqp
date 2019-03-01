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
        print('股票回测研究平台 v0.0.2')
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
            close_price = CStockDaily.get_close(ts_code, current_date)
            if rst[0] > 0:
                print('{0}买入股票'.format(current_date))
                # 获取现金资产值
                # 在t_account_io中转出10%
                # 根据收盘价计算10%金额可以买的股票数，得出实际金额
                # 现金资产减少实际金额
                # 根据股价增加相应持股量
                # 在股票流水表中添加买入记录
                # 增加账户中的股票资产，增加值为实际金额
            else:
                print('{0}卖出股票'.format(current_date))
                # 计算卖出10%的发生金额
                # 减少股票持股量
                # 在股票流水表中添加卖出记录
                # 减少账户中股示资产
                # 在账户流水中增加充入记录，金额为发生金额
                # 增加账户现金资产
            # 将账户信息存入当前日期的历史表
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
        


