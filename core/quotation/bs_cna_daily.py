'''
baostock A股日线行情工具类
'''
import baostock as bs
import pandas as pd

class BsCnaDaily(object):
    def __init__(self):
        self.name = 'BsADaily'
        
    def get_history_data(self, stock_code, start_date, end_date):
        '''
        获取A股日线行情历史数据
        @param stock_code 股票代码，如工商银行为：sh.601398
        @param start_date 开始日期，格式yyyy-MM-dd
        @param end_date 结束日期，格式yyyy-MM-dd
        @return 获取成功返回True，否则返回False
        @version v0.0.1 闫涛 2019-04-22
        '''
        lg = bs.login()
        if lg.error_code != '0':
            print('login respond  error_msg:'+lg.error_msg)
            return False, lg.error_msg
        rs = bs.query_history_k_data_plus(stock_code,
                "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3")
        if rs.error_code != '0':
            print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
            return False,rs.error_msg
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_csv('./data/{0}.csv'.format(stock_code), index=False)
        bs.logout()
        return True