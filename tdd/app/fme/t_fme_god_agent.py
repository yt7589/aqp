import unittest
from app.fme.fme_god_agent import FmeGodAgent

class TFmeGodAgent(unittest.TestCase):
    def test_run001(self):
        agent = FmeGodAgent()
        agent.train()

    def test_run002(self):
        agent = FmeGodAgent()
        agent.test001()
        