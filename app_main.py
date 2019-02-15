from app_registry import appRegistry as ar
from controller.c_stock import CStock
import model.m_mysql as db

def startup():
    print('hello world {0} db={1}'.format(ar.version, ar.rdb['host']))
    ar.caller = 'app_main'
    CStock.get_stocks()
    
if '__main__' == __name__:
    db.init_db_pool() # 初始化数据库连接池
    startup()
    ar.is_stopping = True # 结束程序