from datetime import date
from datetime import timedelta
import numpy as np
from app_registry import appRegistry as ar
from controller.c_stock_daily import CStockDaily
from controller.c_account import CAccount
from ann.stock_daily_svm import StockDailySvm
from util.stock_daily_svm_model_evaluator import StockDailySvmModelEvaluator
from app.ashare.ashare_strategy1 import AshareStrategy1
from controller.c_account import CAccount
from controller.c_stock import CStock
from controller.c_user_stock import CUserStock

class StockBacktest(object):
    def __init__(self):
        self.name = 'StockBacktest'

    def buy_stock(self, user_id, account_id, ts_code, curr_date, buy_vol):
        '''
        在指定日期买入指定股票
        @param ts_code：股票编码
        @param curr_date：指定日期
        @version v0.0.1 闫涛 2019-03-05
        '''
        close_price = float(CStockDaily.get_real_close(ts_code, curr_date))
        close_price = int(close_price * 100)
        cash_amount, _ = CAccount.get_current_amounts(account_id)
        buy_amount = buy_vol * close_price
        print('{0}={1}*{2}'.format(buy_amount, buy_vol, close_price))
        rst = CAccount.withdraw(account_id, buy_amount)
        if not rst:
            return
        # 更新用户现金资产
        CAccount.update_cash_amount(account_id, cash_amount - buy_amount)
        # 增加用户股票持有量
        stock_id = CStock.get_stock_id_by_ts_code(ts_code)
        CUserStock.buy_stock_for_user(user_id, stock_id, buy_vol, close_price, curr_date)
        # 增加股票资产
        hold_vol = CUserStock.get_user_stock_vol(user_id, stock_id)
        CAccount.update_stock_amount(account_id, hold_vol*close_price)
        print('回测引擎之买入股票')

    def sell_stock(self, user_id, account_id, ts_code, trade_date, sell_vol):
        '''
        在指定日期卖出指定股票
        @param user_id：用户编号
        @param account_id：账户编号
        @param ts_code：股票编号
        @param trade_date：交易日期
        @param sell_vol：卖出数量
        '''
        close_price = float(CStockDaily.get_real_close(ts_code, trade_date))
        close_price = int(close_price * 100)
        cash_amount, _ = CAccount.get_current_amounts(account_id)
        sell_amount = sell_vol * close_price
        print('卖出股票：{0}={1}*{2}'.format(sell_amount, sell_vol, close_price))
        rst = CAccount.deposit(account_id, sell_amount)
        if not rst:
            return
        # 更新用户现金资产
        CAccount.update_cash_amount(account_id, cash_amount + sell_amount)
        # 减少用户股票持有量
        stock_id = CStock.get_stock_id_by_ts_code(ts_code)
        CUserStock.sell_stock_for_user(user_id, stock_id, sell_vol, close_price, trade_date)
        # 更新股票资产
        hold_vol = CUserStock.get_user_stock_vol(user_id, stock_id)
        CAccount.update_stock_amount(account_id, hold_vol*close_price)
        print('回测引擎之卖出股票')



    def startup(self):
        print('股票回测研究平台 v0.0.2')
        account_id = 1
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
            close_price = CStockDaily.get_real_close(ts_code, current_date)
            cash_amount, stock_amount = CAccount.get_current_amounts(account_id)
            if rst[0] > 0:
                print('{0}买入股票'.format(current_date))
                # 在t_account_io中转出
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
        


