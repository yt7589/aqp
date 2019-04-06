from app.qh.qh_engine import QhEngine
from app.qh.qh_evaluator import QhEvaluator
    
MODE_RUN = 1
MODE_EVALUATE = 2

mode = MODE_EVALUATE
    
def startup():
    print('长期持有按月调仓策略演示程序 v0.0.1')
    if MODE_RUN == mode:
        # 运行模型回测
        engine = QhEngine()
        engine.startup()
    else:
        print('评价模型性能')
        evaluator = QhEvaluator()
        evaluator.draw_cumulative_returns()
    

'''
CS294-136深度学习金融应用：https://github.com/bellettif?tab=repositories
股票数据（5分钟级）：http://baostock.com
'''    