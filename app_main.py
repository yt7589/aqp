from app_registry import appRegistry as ar
from controller.c_stock import CStock
from controller.c_stock_daily import CStockDaily
from ann.svm import Svm
import model.m_mysql as db
from util.stock_daily_svm_model_evaluator import StockDailySvmModelEvaluator

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
    
    StockDailySvmModelEvaluator.evaluate_model()
    
if '__main__' == __name__:
    db.init_db_pool() # 初始化数据库连接池
    startup()
    ar.is_stopping = True # 结束程序