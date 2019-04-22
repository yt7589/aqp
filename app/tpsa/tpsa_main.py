from core.quotation.bs_cna_daily import BsCnaDaily

class TpsaMain(object):
    def __init__(self):
        self.name = 'TpsaMain'
        
    def startup(self):
        print('A股交易对策略')
        bs_cna_daily = BsCnaDaily()
        bs_cna_daily.get_history_data('sh.601398', '2017-01-01', '2017-12-31')