# 保存程序所需全局变量

class AppRegistry(object):
    def __init__(self):
        self.name = 'AppRegistry'
        self.version = 'v0.0.1'
        self.ts_token = '071d5ccd05ef6df14ea199aa8bc4ca5b807cb60ecc402597399514c1'
        self.rdb = {'host': '127.0.0.1', 'user': 'quant', 'passwd':'Quant2019', 'db':'QuantDb', 'charset':'utf8', 'port':3306}
        self.wdb = self.rdb
        self.is_stopping = False
        
appRegistry = AppRegistry()
        
