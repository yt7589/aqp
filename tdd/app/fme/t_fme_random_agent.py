import unittest
from app.fme.fme_random_agent import FmeRandomAgent

class TFmeRandomAgent(unittest.TestCase):
    def test_run001(self):
        agent = FmeRandomAgent()
        agent.train()