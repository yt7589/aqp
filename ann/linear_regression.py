from __future__ import absolute_import, division, print_function
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')
        
class LinearRegression(object):
    def __init__(self):
        self.name = 'LinearRegression'

    def load_dataset(self):
        print('载入数据集')
        self.dataset_path = keras.utils.get_file('auto-mpg.data',
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
        print(self.dataset_path)
        column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
        raw_dataset = pd.read_csv(self.dataset_path, names=column_names,
                                na_values='?', comment='\t',
                                sep=' ', skipinitialspace=True)
        dataset = raw_dataset.copy()
        print(dataset.tail())
        print(dataset.isna().sum())
        dataset = dataset.dropna()
        origin = dataset.pop('Origin')
        dataset['USA'] = (origin == 1)*1.0
        dataset['Europe'] = (origin == 2)*1.0
        dataset['Japan'] = (origin == 3)*1.0
        print(dataset.tail())
        train_dataset = dataset.sample(frac=0.8,random_state=0)
        test_dataset = dataset.drop(train_dataset.index)
        #sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
        #plt.show()
        train_stats = train_dataset.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()
        print(train_stats)
        train_labels = train_dataset.pop('MPG')
        test_labels = test_dataset.pop('MPG')
        mu_val = train_stats['mean'].as_matrix()
        std_val = train_stats['std'].as_matrix()
        np.save('./work/mu.txt', mu_val)
        np.save('./work/std.txt', std_val)
        print('##### mu:{0}\r\n{1}'.format(type(mu_val), mu_val))
        print('##### mean: {0}\r\n{1}'.format(type(train_stats['mean']), train_stats['mean']))
        print('##### std: {0}\r\n{1}'.format(type(train_stats['std']), train_stats['std']))
        normed_train_data = self.norm(train_dataset, train_stats['mean'], train_stats['std'])
        normed_test_data = self.norm(test_dataset, train_stats['mean'], train_stats['std'])
        return train_dataset, normed_train_data, train_labels, normed_test_data, test_labels

    def train(self):
        print('train the model')
        print(tf.__version__)
        train_dataset, normed_train_data, train_labels, normed_test_data, test_labels = self.load_dataset()
        model = self.build_model(train_dataset)
        print(model.summary())
        example_batch = normed_train_data[:10]
        example_result = model.predict(example_batch)
        print(example_result)
        #
        EPOCHS = 1000
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                            validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
        self.plot_history(history)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())
        self.evaluate(model, normed_test_data, test_labels)
        print('type:{0}; val:{1}'.format(type(normed_test_data), normed_test_data))
        # 保存模型
        model.save("./work/lr000")

    def evaluate(self, model, normed_test_data, test_labels):
        print('evaluate the model')
        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    def predict(self, data):
        print('predict the value')
        model = keras.models.load_model('./work/lr000')
        weights = np.array(model.get_weights())
        print('weights:{0}'.format(weights.shape))
        # keras权值的组织形式：如本例中输入向量为9维，第一层64个神经元，第2层64个神经元，
        # 第3层1个神经元，则其权值形式为：
        # 9*64, 64(第1层bias),64*64,64(第2层bias),64*1,1(输出层bias)
        for lw in weights:
            print('###:{0}'.format(lw.shape))
        rst = model.predict(data)
        print('type:{0}; {1}'.format(type(rst), rst))


    def norm(self, x, mu_val, std_val):
        return (x - mu_val) / std_val

    def build_model(self, train_dataset):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model

    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
        plt.ylim([0,5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
        plt.ylim([0,20])
        plt.legend()
        plt.show()


if '__main__' == __name__:
    lr = LinearRegression()
    #lr.train()
    # 393  27.0, 4, 140.0, 86.0, 2790.0, 15.6, 82, 1.0, 0.0, 0.0
    raw_data = pd.DataFrame({'Cylinders':[4], 'Displacement': [140.0], 
        'Horsepower':[86.0], 'Weight': [2790.0], 
        'Acceleration': [15.6], 'Model Year': [82], 'USA': [1.0],
        'Europe':[0.0], 'Japan':[0.0]}
    )
    mu_val = np.load('mu.txt.npy')
    std_val = np.load('std.txt.npy')
    data = lr.norm(raw_data, mu_val, std_val)
    lr.predict(data)

