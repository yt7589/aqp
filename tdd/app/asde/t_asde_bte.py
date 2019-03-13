import unittest
from app.asde.asde_bte import AsdeBte

class TAsdeBte(unittest.TestCase):
    def test_get_stock_vo(self):
        stock_id, ts_code, start_dt, end_dt = 69, '603912.SH',\
                    '20180101', '20181231'
        asdeBte = AsdeBte()
        stock_vo = asdeBte.get_stock_vo(stock_id, 
                    ts_code, start_dt, end_dt)
        '''
        print('\r\n{0} {1} train:{2}; test:{3}\r\n    '
                    'mu:{4}\r\n    std:{5}'.format(
                        stock_vo['stock_id'],
                        stock_vo['ts_code'],
                        len(stock_vo['train_x']),
                        len(stock_vo['test_x']),
                        stock_vo['mus'],
                        stock_vo['stds']
                    ))
        '''
        for v in stock_vo['train_x']:
            print('归一化：{0} | {1}'.format(v[0], v[1]))
        self.assertTrue(True)

    def test_get_stocks(self):
        asdeBte = AsdeBte()
        start_dt, end_dt = '20180101', '20181231'
        stocks = asdeBte.get_stocks(start_dt, end_dt)
        for stock in stocks:
            print('{0} {1}:{2}'.format(stock['stock_id'], stock['ts_code'], len(stock['train_x'])))