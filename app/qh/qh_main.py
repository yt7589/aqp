from app.qh.qh_engine import QhEngine
    
def startup():   
    print('长期持有按月调仓策略演示程序 v0.0.1')
    engine = QhEngine()
    engine.startup()