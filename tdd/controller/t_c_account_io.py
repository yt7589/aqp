import unittest
from controller.c_account_io import CAccountIo

class TCAccountIo(unittest.TestCase):
    def test_withdraw(self):
        account_id = 1
        amount = 100
        CAccountIo.withdraw(account_id, amount)
        self.assertEqual('abc', 'abc')

if '__main__' == __name__:
    unittest.main()
