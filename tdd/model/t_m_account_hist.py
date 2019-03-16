import unittest
from model.m_account_hist import MAccountHist
from util.app_util import AppUtil

class TMAccountHist(unittest.TestCase):
    def test_insert_account_hist(self):
        account_id = 1
        account_date = AppUtil.parse_date('2019-03-16', AppUtil.DF_HYPHEN)
        cash_amount = 88
        stock_amount = 99
        pk, affected_rows = MAccountHist.insert_account_hist(
            account_id,
            account_date,
            cash_amount,
            stock_amount
        )
        print('pk={0}, ar={1}'.format(pk, affected_rows))
        self.assertTrue(True)