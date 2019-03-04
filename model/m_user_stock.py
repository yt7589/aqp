import model.m_mysql as db

'''
管理用户持有股票的信息
'''
class MUserAccount(object):
    def __init__(self):
        self.name = 'MUserAccount'

    @staticmethod
    def update_user_stock(user_id, stock_id, vol):
        '''
        更新用户持股信息，股票价格取前一天收盘价格
        '''