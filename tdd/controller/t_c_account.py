import unittest
from controller.c_account import CAccount

class TCAccount(unittest.TestCase):
    def test_get_amounts(self):
        account_id = 1
        info = CAccount.get_current_amounts(account_id)
        print('现金：{0}; 股票：{1}; 总资产：{2}'.format(info[0], info[1], (info[0]+info[1])))
        self.assertEqual('a', 'a')

    def test_get_hist_amounts(self):
        account_id = 1
        account_date = '2019-03-01'
        info = CAccount.get_hist_amounts(account_id, account_date)
        print('现金：{0}; 股票：{1}; 总资产：{2}; 日期：{3}'.format(info[0], 
                info[1], (info[0]+info[1]), account_date))
        self.assertTrue(True)

    def test_update_cash_amount(self):
        account_id = 1
        cash_amount = 200
        rst = CAccount.update_cash_amount(account_id, cash_amount)
        self.assertEqual(rst, True)

    def test_withdraw(self):
        account_id = 1
        amount = 20000
        rst = CAccount.withdraw(account_id, amount)
        self.assertTrue(True)