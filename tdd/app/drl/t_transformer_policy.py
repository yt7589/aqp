import unittest
import gym
from app.drl.transformer_policy import TransformerPolicy

class TTransformerPolicy(unittest.TestCase):
    def test_startup(self):
        tp = TransformerPolicy()
        tp.startup()
        self.assertTrue(True)