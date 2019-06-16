import unittest
from app.fme.fme_engine import FmeEngine

class TFmeEngine(unittest.TestCase):
    def test_startup(self):
        fme_engine = FmeEngine()
        fme_engine.startup()