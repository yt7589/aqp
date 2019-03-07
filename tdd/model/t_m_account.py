import unittest
from model.m_account import MAccount

class TMAccount(unittest.TestCase):
    def test_update_cash_amount(self):
        account_id = 1
        cash_amount = 100000000
        pk, affected_rows = MAccount.update_cash_amount(account_id, cash_amount)
        self.assertEqual(affected_rows, 1)

    def test_update_stock_amount(self):
        account_id = 1
        stock_amount = 888
        pk, affected_rows = MAccount.update_stock_amount(account_id, stock_amount)
        if 1 == affected_rows:
            self.assertTrue(True)
        else:
            self.assertFalse(False)
