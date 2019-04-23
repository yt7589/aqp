from core.quotation.bs_cna_daily import BsCnaDaily

class TpsaMain(object):
    def __init__(self):
        self.name = 'TpsaMain'
        
    def startup(self):
        print('A股交易对策略')
        # self.get_quotation_data() # 获取行情数据
        
    def get_quotation_data(self):
        # 取工商银行2016-10-27~2019-04-23
        bs_cna_daily = BsCnaDaily()
        # 工商银行（股票代码60开头的是上海）
        bs_cna_daily.get_history_data('sh.601398', 
                    '2008-01-01', '2019-04-23')
        # 建设银行
        bs_cna_daily.get_history_data('sh.601939', 
                    '2008-01-01', '2019-04-23')
        # 浦发银行
        bs_cna_daily.get_history_data('sh.600000', 
                    '2008-01-01', '2019-04-23')