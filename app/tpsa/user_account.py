# 用户账户类，管理用户现金账户equity，equity0记录初始值
# etfs账户为一个数组，格式为：
# etfs = [
#    {'name': 'ICBC', 'volume': 1000, 'amount': 2999.00},
#    ......
# ]
# 对应的行情文件为：./data/ICBC.csv，
# 每个股票的金额为数量乘以最后的收盘价，其初始值均为0
import threading

class UserAccount(object):
    def __init__(self, equity, etf_names):
        '''
        用于管理用户的现金和股票账户情况
        @param equity 初始金额，单位为元
        @param etfs_name 股票列表，格式['ICBC', 'CBC']
        '''
        self.name = 'UserAccount'
        self.equity = equity
        self.equity0 = self.equity
        self.etfs = []
        for etf_name in etf_names:
            etf = {'name': etf_name, 'volume': 0, 'amount': 0}
            self.etfs.append(etf)
        # 当一个策略要操作用户账户时，必须按如下方式操作：
        # self.lock.acquire()
        # ...... 操作账户信息 ......
        # self.lock.release()
        self.lock = threading.Lock()