import numpy as np

class CAccountIo(object):
    def __init__(self):
        self.name = 'CAccountIo'

    @staticmethod
    def withdraw(account_id, amount):
        '''
        从指定账户取出指定金额，增加资金转出金额记录，并更新用户现金资产
        '''
        print('CAccountIo.withdraw')