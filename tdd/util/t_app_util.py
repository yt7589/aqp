import unittest
from util.app_util import AppUtil

class TAppUtil(unittest.TestCase):
    def test_get_delta_date(self):
        curr_date = '20190305'
        delta = 3
        df = AppUtil.DF_COMPACT
        dt = AppUtil.get_delta_date(curr_date, delta, df)
        print('日期为：{0}'.format(dt))
        self.assertEqual('20190308', dt)

    def test_get_current_date_str(self):
        dt = AppUtil.get_current_date_str()
        print('当前日期：{0}'.format(dt))
        self.assertTrue(True)

    def test_change_date_compact_to_hyphen(self):
        dt = '20190102'
        dt1 = AppUtil.change_date_compact_to_hyphen(dt)
        print('新格式：{0}'.format(dt1))
        self.assertTrue('2019-01-02'==dt1)

    def test_parse_date(self):
        dt = '20190101'
        dt1 = AppUtil.parse_date(dt)
        print('type:{0} --- {1}'.format(type(dt1), AppUtil.format_date(dt1)))