import unittest
from model.m_user import MUser

class TMUser(unittest.TestCase):
    def test_get_user_id_of_account_id(self):
        account_id = 1
        rc, rows = MUser.get_user_id_of_account_id(account_id)
        print(rows)
        self.assertTrue(True)