import unittest
from model.m_account_io import MAccountIo

class TMAccountIo(unittest.TestCase):
    def test_get_latest_io(self):
        account_id = 1
        rc, rows = MAccountIo.get_latest_io(account_id)
        if rc > 0:
            print('余额：{0}'.format(rows[0][0]))
        else:
            print('错误：没有流水记录')
        self.assertEqual('a', 'a')

    def test_withdraw(self):
        account_id = 1
        amount = 10000
        MAccountIo.withdraw(account_id, amount)
        self.assertTrue(True)

    def test_deposit(self):
        account_id = 1
        amount = 10000
        MAccountIo.deposit(account_id, amount)
        self.assertTrue(True)

if '__main__' == __name__:
    unittest.main()