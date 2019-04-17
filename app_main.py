import sys
sys.path.append('./core')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#import tensorflow as tf
from app_registry import appRegistry as ar
#import model.m_mysql as db
import app.qh.qh_main as qh
import app.pqb.pqb_main as pqb



def call_stock_backtest():
    stock_backtest = StockBacktest()
    stock_backtest.startup()

def startup():
    
    qh.startup()
    #pqb.startup()
    
    
    
    #call_stock_backtest()
    #test_get_amounts()


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
    startup()