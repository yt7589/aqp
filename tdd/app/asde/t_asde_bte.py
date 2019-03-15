import unittest
from datetime import date
from datetime import timedelta
from app.asde.asde_bte import AsdeBte
from app.asde.strategy.asde_strategy_1 import AsdeStrategy1

class TAsdeBte(unittest.TestCase):
    def test_startup(self):
        asde_bte = AsdeBte()
        asde_bte.startup()
        self.assertTrue(True)

    def test_run_engine(self):
        user_id = 1
        account_id = 1
        backtest_date = date(2019, 1, 1)
        asde_bte = AsdeBte()
        stocks = asde_bte.get_stocks('20180101', '20181231')
        asde_bte.run_engine(user_id, account_id, stocks, backtest_date)


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

    def test_process_stocks_daily(self):
        user_id = 1
        account_id = 1
        asde_bte = AsdeBte()
        asde_bte.strategy = AsdeStrategy1()
        stocks = asde_bte.get_stocks('20180101', '20181231')
        for stock in stocks:
            asde_bte.strategy.setup_stock_ml_model(stock)
        asde_bte.process_stocks_daily(user_id, account_id, stocks, '20190101')