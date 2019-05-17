#
import unittest
import numpy as np
import tensorflow as tf
from app.drl.probability_distribution import ProbabilityDistribution

class TProbabilityDistribution(unittest.TestCase):
    def test_call(self):
        print('测试ProbabilityDistribution...')
        pd = ProbabilityDistribution()
        logits = tf.log([[1., 900., 1., 500., 1., 1., 1.]])
        for i in range(10):
            a1 = pd.predict(logits, steps=1)
            print('action={0}'.format(a1[0]))
        self.assertTrue(True)