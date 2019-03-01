import numpy as np
import tensorflow as tf
from app_registry import appRegistry as ar
from controller.c_stock import CStock
from controller.c_stock_daily import CStockDaily
from ann.svm import Svm
import model.m_mysql as db
from util.stock_daily_svm_model_evaluator import StockDailySvmModelEvaluator
from controller.c_portfolio import CPortfolio
from ann.linear_regression_engine import LinearRegressionEngine
from app.kelly_demo import KellyDemo
from util.learn import Learn
from app.stock_backtest import StockBacktest
from controller.c_account import CAccount

def call_stock_backtest():
    #stock_backtest = StockBacktest()
    #stock_backtest.startup()
    close = CStockDaily.get_close('603912.SH', '20190103')
    print('收盘价：{0}'.format(close))

def test_get_amounts():
    account_id = 1
    info = CAccount.get_current_amounts(account_id)
    print('现金：{0}; 股票：{1}; 总资产：{2}'.format(info[0], info[1], (info[0]+info[1])))

def test_get_hist_amounts():
    account_id = 1
    account_date = '2019-03-01'
    info = CAccount.get_hist_amounts(account_id, account_date)
    print('现金：{0}; 股票：{1}; 总资产：{2}; 日期：{3}'.format(info[0], 
            info[1], (info[0]+info[1]), account_date))

def startup():
    #call_stock_backtest()
    #test_get_amounts()
    test_get_hist_amounts()


    #CStock.get_stocks()
    #CStockDaily.get_stock_daily_kline(1, '20180101', '20190214')
    #CStockDaily.generate_stock_daily_ds('603912.SH', '20180101', '20190213')
    '''
    model = Svm.train(train_x, train_y, validate_x, validate_y)
    rst = Svm.predict(model, test_x)
    print('rst={0}'.format(rst))
    '''
    
    '''
    Svm.train()
    rst = Svm.predict(CStockDaily.test_x)
    print('svm_rst:{0}'.format(rst))
    '''
    # ts_code = '603912.SH'
    # StockDailySvmModelEvaluator.evaluate_model(ts_code)
    
    # 投资组合
    '''
    ts_codes = ['603912.SH', '300666.SZ', '300618.SZ', '002049.SZ', '300672.SZ']
    start_dt = '20180101'
    duration = 90
    CPortfolio.get_portfolio(ts_codes, start_dt, duration)
    '''

    #lre = LinearRegressionEngine()
    #lre.startup()

    '''
    learn = Learn()
    learn.startup()
    '''


    
    
if '__main__' == __name__:
    db.init_db_pool() # 初始化数据库连接池
    startup()
    ar.is_stopping = True # 结束程序