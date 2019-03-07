import time
import datetime
from datetime import date

class AppUtil(object):
    DF_COMPACT = '%Y%m%d' # 日期格式为：20190305
    DF_HYPHEN = '%Y-%m-%d' # 日期格式为：2019-03-05

    def __init__(self):
        self.name = 'AppUtil'

    @staticmethod
    def get_delta_date(curr_date, delta, df=DF_COMPACT):
        '''
        获取指定日期前或后几天字符串
        @param curr_date：指定日期，格式为20190305
        @param delta：1为明天，-1为昨天
        @param df：日期格式，可以取值为DF_COMPACT或DF_HYPHEN
        @return 指定日期字符串，格式为20190306
        @version v0.0.1 闫涛 2019-03-05
        '''
        curr_date_obj = datetime.datetime.strptime(curr_date, df)
        prev_date_obj = curr_date_obj + datetime.timedelta(days=delta)
        return datetime.date.strftime(prev_date_obj, df)

    @staticmethod
    def get_current_date_str(df=DF_COMPACT):
        '''
        获取当前日期字符串，
        @param df：格式为DF_COMPACT或DF_HYPHEN
        @return 日期字符串
        @version v0.0.1 闫涛 2019-03-05
        '''
        return time.strftime(df, time.localtime())

    @staticmethod
    def change_date_compact_to_hyphen(dt):
        '''
        将20190102格式的日期转变为2019-01-02格式日期字符串
        @param dt：20190102格式日期字符串
        @return 2019-01-02格式字符串
        @version v0.0.1 闫涛 2019-03-07
        '''
        date_obj = datetime.datetime.strptime(dt, AppUtil.DF_COMPACT)
        return datetime.date.strftime(date_obj, AppUtil.DF_HYPHEN)