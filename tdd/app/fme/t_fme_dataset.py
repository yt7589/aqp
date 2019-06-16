import unittest
import numpy as np
import tensorflow as tf
from app.fme.fme_dataset import FmeDataset

class TFmeDataset(unittest.TestCase):
    def test_create_exp_dataset(self):
        fme_dataset = FmeDataset()
        fme_dataset.create_exp_dataset()

    def test_serialize_example(self):
        fme_dataset = FmeDataset()
        x = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        label = 1
        rec = fme_dataset.serialize_example(x.tobytes(), label)
        print(rec)
        rst = tf.train.Example.FromString(rec)
        print(rst)

    def test_create_bitcoin_dataset(self):
        fme_dataset = FmeDataset()
        fme_dataset.create_bitcoin_dataset(dataset_size=500)

    def test_load_bitcoin_dataset(self):
        fme_dataset = FmeDataset()
        X, y = fme_dataset.load_bitcoin_dataset()
        print('X:{0}; {1}'.format(X.shape, X))
        print('y:{0}; {1}'.format(y.shape, y))