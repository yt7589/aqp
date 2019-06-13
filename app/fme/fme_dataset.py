import numpy as np
import tensorflow as tf


class FmeDataset(object):
    FME_LABELS = {
        'buy': 0,
        'sell': 1,
        'hold': 2
    }
    fme_features = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=2),
    }

    def __init__(self):
        self.name = 'FmeDataset'

    def create_np_dataset(self, dataset_size = 60000):
        a1 = np.random.randn(dataset_size)
        c1 = 1.1
        a2 = np.random.randn(dataset_size)
        c2 = 2.1
        a3 = np.random.randn(dataset_size)
        c3 = 3.2
        a4 = np.random.randn(dataset_size)
        c4 = 4.3
        a5 = np.random.randn(dataset_size)
        c5 = 5.1
        delta = np.random.randn(dataset_size)
        a6_ = a1*c1 + a2*c2 + a3*c3 + a4*c4 + a5*c5
        a6 = a6_ + delta
        y_ = np.zeros((dataset_size), dtype=np.int32)
        for i in range(len(a6)):
            if a6[i] - a6_[i] > 0.3:
                y_[i] = 0
            elif a6[i] - a6_[i] < -0.3:
                y_[i] = 2
            else:
                y_[i] = 1
        X = np.reshape(np.array([a1, a2, a3, a4, a5, a6]).T, (dataset_size, 6))
        y = y_.T
        return X, y

    def create_exp_dataset(self, dataset_size = 60000, 
                tfr_file = 'train_dataset.tfrecords'):
        a1 = np.random.randn(dataset_size)
        c1 = 1.1
        a2 = np.random.randn(dataset_size)
        c2 = 2.1
        a3 = np.random.randn(dataset_size)
        c3 = 3.2
        a4 = np.random.randn(dataset_size)
        c4 = 4.3
        a5 = np.random.randn(dataset_size)
        c5 = 5.1
        delta = np.random.randn(dataset_size)
        a6_ = a1*c1 + a2*c2 + a3*c3 + a4*c4 + a5*c5
        a6 = a6_ + delta
        y = np.zeros((dataset_size), dtype=np.int32)
        for i in range(len(a6)):
            if a6[i] - a6_[i] > 0.3:
                y[i] = 0
            elif a6[i] - a6_[i] < -0.3:
                y[i] = 2
            else:
                y[i] = 1
        X_ = np.reshape(np.array([a1, a2, a3, a4, a5, a6]).T, (dataset_size, 6, 1))
        X = tf.data.Dataset.from_tensor_slices(tf.cast(X_, tf.float64))
        print('X:{0}'.format(X))
        y_ = y.T
        y = tf.data.Dataset.from_tensor_slices(tf.cast(y_, tf.int64))
        print('y:{0}'.format(y))
        ds = tf.data.Dataset.from_tensor_slices((X_, y_))
        print(ds)


    def serialize_example(self, data, label):
        feature = {
            'data': FmeDataset._bytes_feature(data),
            'label': FmeDataset._int64_feature(label)
        }
        return tf.train.Example(features=tf.train.Features(
                    feature=feature)).SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



        