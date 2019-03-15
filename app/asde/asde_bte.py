# A股日线环境回测引擎
import time
from datetime import date
from datetime import timedelta
import numpy as np
import app_registry
from app_registry import appRegistry as ar
from controller.c_stock_daily import CStockDaily
from controller.c_account import CAccount
from ann.stock_daily_svm import StockDailySvm
#from util.stock_daily_svm_model_evaluator import StockDailySvmModelEvaluator
from app.ashare.ashare_strategy1 import AshareStrategy1
from controller.c_account import CAccount
from controller.c_stock import CStock
from controller.c_user_stock import CUserStock
from app.asde.asde_ds import AsdeDs
from app.asde.ml.asde_svm import AsdeSvm
from util.app_util import AppUtil
from app.asde.strategy.asde_strategy_1 import AsdeStrategy1

class AsdeBte(object):
    def startup(self):
        print('A股日线回测研究平台 v0.0.1')
        account_id = 1
        # 求出已知数据均值和方差
        start_dt = '20180101'
        end_dt = '20181231'
        # 初始化策略
        self.strategy = AsdeStrategy1()
        # 获取股票池
        stocks = self.get_stocks(start_dt, end_dt)
        # 训练初始模型
        for stock in stocks:
            self.strategy.setup_stock_ml_model(stock)
        # 开始进行回测
        backtest_date = AppUtil.parse_date('20190101') 
        # 运行回测引擎
        self.run_engine(account_id, stocks, backtest_date)

    def run_engine(self, account_id, stocks, start_date):
        '''
        从开始日期开始，一直到当前日期为止
        @param start_date：开始回测日期
        @version v0.0.1 闫涛 2019-03-14
        '''
        BT_DAYS = 50000
        backtest_date = start_date
        today = date.today() + timedelta(days=-1)
        for i in range(BT_DAYS):
            if today <= backtest_date:
                break
            # 求出当天的收收盘价，送到策略类中进行预测
            # 求出next_date收盘价作为交易价格
            backtest_date = self.process_stocks_daily(account_id, stocks, backtest_date)

    def process_stocks_daily(self, account_id, stocks, backtest_date):
        '''
        求出backtest_date的行情数据，传递给策略类（历史数据由策略类自己维护），策略类根据业务
        逻辑，返回买入卖出每支股票的数量，获取下一个交易日next_date的收盘价，执行相应的交易，
        @param backtest_date：必须是具有行情数据的日期，由调用者保证
        @return 下一个具有行情的交易日，执行交易的日期
        @version v0.0.1 闫涛 2019-03-15
        '''
        for stock in stocks:
            trade_date, quotation = CStockDaily.get_daily_quotation(
                        stocks['ts_code'], backtest_date)
            # 传给策略类做决策
            direction, shares = self.strategy.run(account_id, stock, trade_date, quotation)
            if app_registry.ASDE_BTE_BUY == direction:
                # 买入指定数量股票
                pass
            if app_registry.ASDE_BTE_SELL == direction:
                # 卖出指定数量股票
                pass
        next_date = None
        return next_date

    def get_stocks(self, start_dt, end_dt):
        '''
        获取股票池中股票的基本信息、均值、方差、训练样本集、验证样本集和测试样本集
        @param start_dt：开始时间
        @param end_dt：结束时间
        @return 股票池中股票信息列表
        @version v0.0.1 闫涛 2019-03-12
        '''
        stocks = []
        stock_vo = self.get_stock_vo(69, '603912.SH', start_dt, end_dt)
        stocks.append(stock_vo)
        #stock_vo = self.get_stock_vo(1569, '300666.SZ', start_dt, end_dt)
        #stocks.append(stock_vo)
        return stocks
    
    def get_stock_vo(self, stock_id, ts_code, start_dt, end_dt):
        '''
        获取单支股票的基本信息和数据集
        @param stock：股票编号
        @param ts_code；股票编码
        @param start_dt：开始日期
        @param end_dt：结束日期
        @return 返回股票基本信息，均值、方差和训练样本集、验证样本集、测试样本集
        @version v0.0.1 闫涛 2019.03.12
        '''
        stock_vo = {'stock_id': stock_id, 'ts_code': ts_code}
        # 求出均值和方差
        stock_vo['train_x'], stock_vo['train_y'], \
                    stock_vo['validate_x'], stock_vo['validate_y'], \
                    stock_vo['test_x'] = CStockDaily.\
                        generate_stock_daily_ds(ts_code, start_dt, end_dt)
        stock_vo['mus'], stock_vo['stds'] = AsdeDs.get_mean_stds(stock_vo['train_x'])
        # 对原始数据集进行归一化
        AsdeDs.normalize_datas(stock_vo['train_x'], stock_vo['mus'], stock_vo['stds'])
        return stock_vo






    # **********************************************************************
    # **********************************************************************
    # **********************************************************************
    # **********************************************************************
    # **********************************************************************
    # **********************************************************************

    
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