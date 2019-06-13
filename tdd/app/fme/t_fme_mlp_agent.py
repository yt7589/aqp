import unittest
from app.fme.fme_mlp_agent import FmeMlpAgent

class TFmeMlpAgent(unittest.TestCase):
    def test_generate_dataset001(self):
        agent = FmeMlpAgent()
        agent.generate_dataset()
        