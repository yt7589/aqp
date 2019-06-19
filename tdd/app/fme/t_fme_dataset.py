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

    def test_bug(self):
        fme_dataset = FmeDataset()
        X, y = fme_dataset.load_bitcoin_dataset()
        print('X:{0}'.format(X.shape))
        self.print_ds_rec(X[0])
        self.print_ds_rec(X[1])
        self.print_ds_rec(X[2])
        self.print_ds_rec(X[3])
        self.print_ds_rec(X[4])
        self.print_ds_rec(X[5])
        #
        idx = 6
        self.print_ds_rec(X[idx])
        idx += 1
        self.print_ds_rec(X[idx])
        idx += 1
        self.print_ds_rec(X[idx])
        idx += 1
        self.print_ds_rec(X[idx])
        idx += 1
        self.print_ds_rec(X[idx])
        idx += 1
        self.print_ds_rec(X[idx])
        idx += 1
        print('*************')
        self.time_span = 5
        self.commission = 0.007
        btc_held = 1

        #
        tidx = 9
        action = self._choose_action(btc_held, X, tidx, 5)
        print('tidx={0}; action={1}'.format(tidx, action))
        #
        tidx = 10
        action = self._choose_action(btc_held, X, tidx, 5)
        print('tidx={0}; action={1}'.format(tidx, action))
        #
        tidx = 11
        action = self._choose_action(btc_held, X, tidx, 5)
        print('tidx={0}; action={1}'.format(tidx, action))
        #
        tidx = 12
        action = self._choose_action(btc_held, X, tidx, 5)
        print('tidx={0}; action={1}'.format(tidx, action))

    def print_ds_rec(self, ds_rec):
        print('{0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f} || '
                '{5:0.3f} {6:0.3f} {7:0.3f} {8:0.3f} {9:0.3f} || '
                '{10:0.3f} {11:0.3f} {12:0.3f} {13:0.3f} {14:0.3f} || '
                '{15:0.3f} {16:0.3f} {17:0.3f} {18:0.3f} {19:0.3f} || '
                '{20:0.3f} {21:0.3f} {22:0.3f} {23:0.3f} {24:0.3f} => '
                '{25:0.1f}'.format(
            ds_rec[0], ds_rec[1], ds_rec[2], ds_rec[3], ds_rec[4],
            ds_rec[5], ds_rec[6], ds_rec[7], ds_rec[8], ds_rec[9],
            ds_rec[10], ds_rec[11], ds_rec[12], ds_rec[13], ds_rec[14],
            ds_rec[15], ds_rec[16], ds_rec[17], ds_rec[18], ds_rec[19],
            ds_rec[20], ds_rec[21], ds_rec[22], ds_rec[23], ds_rec[24],
            ds_rec[25]
        ))

    def _choose_action(self, btc_held, X, idx, frame_size):
        current_price = X[idx, 23]
        future_prices = X[idx+1:idx+self.time_span+1, (frame_size-1)*5+3:(frame_size-1)*5+4]
        commission = self.commission + 0.001
        print('current_price:{0}; future_prices:{1}'.format(current_price, future_prices))
        action = 2.0
        for price in future_prices:
            price_delta = (price - current_price) / current_price
            if price_delta < -commission: # 跌幅过大
                if 1 == btc_held:
                    action = 1.0
                    break
                else:
                    break
            elif price_delta > commission: # 涨幅过大
                if 0 == btc_held:
                    action = 0.0
                    break
                else:
                    break
        return action
