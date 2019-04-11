# 保存程序所需全局变量

ASDE_BTE_BUY = 1001 # 买入股票
ASDE_BTE_SELL = 1002 # 卖出股票

class AppRegistry(object):
    def __init__(self):
        self.name = 'AppRegistry'
        self.in_colab = True
        self.version = 'v0.0.1'
        self.ts_token = '071d5ccd05ef6df14ea199aa8bc4ca5b807cb60ecc402597399514c1'
        if not self.in_colab:
            self.rdb = {'host': '127.0.0.1', 'user': 'quant', 'passwd':'Quant2019', 'db':'QuantDb', 'charset':'utf8', 'port':3306}
            self.wdb = self.rdb
        self.is_stopping = False
        self.increase_threshold = 1.0
        
appRegistry = AppRegistry()
        
