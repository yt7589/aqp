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