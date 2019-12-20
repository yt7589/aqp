#
import unittest
from app.tf2.drl_base import DrlBase

class TDrlBase(unittest.TestCase):
    def test_startup(self):
        drl_base = DrlBase()
        drl_base.startup()