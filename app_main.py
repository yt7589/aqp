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

def startup():
    print('hello world {0} db={1}'.format(ar.version, ar.rdb['host']))
    ar.caller = 'app_main'
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

    learn = Learn()
    learn.startup()


    
    
if '__main__' == __name__:
    db.init_db_pool() # 初始化数据库连接池
    startup()
    ar.is_stopping = True # 结束程序