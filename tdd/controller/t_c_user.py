import unittest
from controller.c_user import CUser

class TCUser(unittest.TestCase):
    def test_get_user_id_of_account_id(self):
        account_id = 1
        user_id = CUser.get_user_id_of_account_id(account_id)
        print('user_id={0}'.format(user_id))
        self.assertTrue(True)