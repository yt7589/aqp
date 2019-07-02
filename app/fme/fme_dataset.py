import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing


class FmeDataset(object):
    DATASET_SIZE = 1000
    
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
        self.quotation_file = './data/bitstamp.csv'
        self.ds_max_min = './work/btc_max_min.csv'
        self.ds_x_file = './work/btc_x_1.csv'
        self.ds_y_file = './work/btc_y_1.csv'
        self.commission = 0.007
        self.time_span = 15
        self.frame_size = 5

    def load_bitcoin_dataset(self):
        X = np.loadtxt(self.ds_x_file, delimiter=',')
        y = np.loadtxt(self.ds_y_file, delimiter=',')
        return X, y

    def create_bitcoin_dataset(self, dataset_size=1000, frame_size=5):
        ''' 根据比特币交易数据生成系列数据集，每个数据集缺省包括1000个时间点 '''
        print('bitcoin dataset generation...')
        dataset_size += self.time_span
        # 读入比特币行情
        df = pd.read_csv(self.quotation_file)
        df = df.sort_values('Timestamp')
        df = df.dropna().reset_index()
        scaler = preprocessing.MinMaxScaler() # 将数值归一化到0~1
        start_idx = 0
        end_idx = start_idx + dataset_size
        # 读入归一化的比特币行情数据
        scaled_df = df.values[start_idx:end_idx].astype(np.float64)
        max_min = self.get_quotation_max_min(scaled_df) # 求出最大、最小值
        np.savetxt(self.ds_max_min, max_min, delimiter = ',')
        scaled_df = scaler.fit_transform(scaled_df)
        scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
        X_raw = np.array([
            scaled_df['Open'].values[:],
            scaled_df['High'].values[:],
            scaled_df['Low'].values[:],
            scaled_df['Close'].values[:],
            scaled_df['Volume_(BTC)'].values[:],
        ])
        X_raw = X_raw.T
        # 将之前5个时间点的行情数据作为一个训练样本
        self.frame_size = frame_size
        X_train = np.array([])
        for idx in range(frame_size-1, X_raw.shape[0] - self.time_span):
            # 形成frame
            for i in range(idx-frame_size+1, idx+1):
                X_train = np.append(X_train, X_raw[i])
        X_train = X_train.reshape((
            X_raw.shape[0] - self.time_span - frame_size + 1, 
            X_raw.shape[1] * frame_size
        ))
        # 求出持有比特币时的最佳行动作为标签
        y_train = np.full((X_train.shape[0], ), 2.0)
        btc_held = 1
        for idx in range(y_train.shape[0]):
            #y_train[idx] = self._choose_action(btc_held, X_raw, idx, frame_size)
            y_train[idx] = self._choose_action(btc_held, X_train, idx, frame_size)
        # 复制样本集，求出在没有持有比特币时的最佳行动并作为标签
        X_train1 = np.array(X_train, copy=True)
        y_train1 = np.array(y_train, copy=True)
        btc_held = 0
        for idx in range(y_train1.shape[0]):
            #y_train1[idx] = self._choose_action(btc_held, X_raw, idx, frame_size)
            y_train1[idx] = self._choose_action(btc_held, X_train1, idx, frame_size)
        # 在训练样本最后加入一列，分别代表比特币仓位情况：1满仓；0空仓
        c1 = np.ones(X_train.shape[0])
        X_train = np.c_[X_train, c1]
        c0 = np.zeros(X_train1.shape[0])
        X_train1 = np.c_[X_train1, c0]
        # 将满仓和空仓数据集进行合并
        X_train = np.append(X_train, X_train1, axis=0)
        y_train = np.append(y_train, y_train1, axis=0)
        # 将数据集保存为文件
        np.savetxt(self.ds_x_file, X_train, delimiter = ',')
        np.savetxt(self.ds_y_file, y_train, delimiter=',')
        return X_train, y_train

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

    def get_quotation_max_min(self, ds):
        ''' 
        求出行情段中开盘价、最高价、最低价、收盘价、交易量的最大值的
        最小值，并以2*5数组形式返回
        '''
        raw = ds[:, 2:7]
        nd_max = np.amax(raw, axis=0)
        nd_min = np.amin(raw, axis=0)
        return np.append([nd_max], [nd_min], axis=0)



        